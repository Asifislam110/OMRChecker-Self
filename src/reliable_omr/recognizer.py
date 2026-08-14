"""Public library API for schema-driven reliable OMR recognition."""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from src.reliable_omr.bubbles import recognize_answers, recognize_roll_number
from src.reliable_omr.calibration import (
    SerializedRiskCalibrator,
    build_risk_features,
    route_risk,
)
from src.reliable_omr.definition import DefinitionInput, load_sheet_definition
from src.reliable_omr.opencv import require_cv2
from src.reliable_omr.preview import (
    DEFAULT_PREVIEW_MAX_HEIGHT,
    DEFAULT_PREVIEW_MAX_JPEG_BYTES,
    DEFAULT_PREVIEW_MAX_WIDTH,
    encode_rectified_preview,
)
from src.reliable_omr.quality import (
    CAPTURE_MODES,
    add_marker_quality,
    assess_input_quality,
)
from src.reliable_omr.rectification import rectify_to_canonical
from src.reliable_omr.types import (
    DocumentResult,
    MarkStatus,
    QualityIssue,
    ReviewReason,
    SheetResult,
    SheetStatus,
)


PROCESSOR_VERSION = "reliable-omr/1.1.0"


class RecognitionInputError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image or PDF."""


class QRPayloadError(ValueError):
    """Raised when a decoded QR value violates its sheet contract."""

    def __init__(self, code: str, message: str, scope: str = "sheet.qr"):
        super().__init__(message)
        self.code = code
        self.scope = scope


_BASE64URL = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_qr_payload(
    raw: str, definition: Mapping[str, Any]
) -> Dict[str, Any]:
    """Parse O1/O2 syntax only; O2 identity/HMAC verification is external."""

    qr_definition = definition["qr"]
    prefix = qr_definition["prefix"]
    if not isinstance(raw, str) or not raw.startswith(prefix):
        raise QRPayloadError(
            "qr_prefix_mismatch",
            "QR payload prefix does not match the sheet definition",
        )
    try:
        raw_bytes = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise QRPayloadError(
            "qr_encoding_invalid", "QR payload must contain ASCII only"
        ) from exc

    encoded_payload = raw[len(prefix) :]
    field_names = [
        field["name"] for field in qr_definition["payload_fields"]
    ]
    if prefix == "O2:":
        if len(raw_bytes) != 32:
            raise QRPayloadError(
                "qr_o2_length_invalid",
                "O2 QR payload must be exactly 32 ASCII bytes",
            )
        values = encoded_payload.split(".")
        if (
            len(values) != 2
            or len(values[0]) != 22
            or len(values[1]) != 6
            or any(_BASE64URL.fullmatch(value) is None for value in values)
        ):
            raise QRPayloadError(
                "qr_o2_syntax_invalid",
                "O2 QR payload must contain a 22-character profile token "
                "and 6-character signature in unpadded base64url",
            )
        payload: Any = {
            "profile_token": values[0],
            "signature": values[1],
        }
    else:
        try:
            if encoded_payload.lstrip().startswith("{"):
                payload = json.loads(encoded_payload)
            elif field_names == [
                "exam_id",
                "template_version",
                "template_checksum",
            ]:
                values = encoded_payload.split(".")
                if len(values) != len(field_names) or any(
                    not value for value in values
                ):
                    raise ValueError("invalid compact OMR identity")
                payload = dict(zip(field_names, values))
            else:
                payload = json.loads(encoded_payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise QRPayloadError(
                "qr_json_invalid",
                "QR payload does not match its configured encoding",
            ) from exc

    if not isinstance(payload, dict):
        raise QRPayloadError(
            "qr_payload_not_object", "QR payload must be a JSON object"
        )
    unexpected = sorted(set(payload) - set(field_names))
    if unexpected:
        raise QRPayloadError(
            "qr_unexpected_field",
            "QR payload contains fields not declared by the definition: "
            "{}".format(unexpected),
        )
    for field in qr_definition["payload_fields"]:
        name = field["name"]
        value = payload.get(name)
        if field["required"] and value is None:
            raise QRPayloadError(
                "qr_required_field_missing",
                "QR field '{}' is required".format(name),
                "sheet.qr.{}".format(name),
            )
        if value is None:
            continue
        expected = str if field["type"] == "string" else int
        if not isinstance(value, expected) or (
            expected is int and isinstance(value, bool)
        ):
            raise QRPayloadError(
                "qr_field_type_invalid",
                "QR field '{}' has the wrong type".format(name),
                "sheet.qr.{}".format(name),
            )
        length = len(str(value))
        if length < field.get("min_length", 0):
            raise QRPayloadError(
                "qr_field_too_short",
                "QR field '{}' is below its minimum length".format(name),
                "sheet.qr.{}".format(name),
            )
        if length > field.get("max_length", length):
            raise QRPayloadError(
                "qr_field_too_long",
                "QR field '{}' exceeds its maximum length".format(name),
                "sheet.qr.{}".format(name),
            )
    if (
        "sheet_definition_id" in field_names
        and payload.get("sheet_definition_id") != definition["definition_id"]
    ):
        raise QRPayloadError(
            "qr_definition_mismatch",
            "QR sheet_definition_id does not match the supplied definition",
            "sheet.qr.sheet_definition_id",
        )
    return payload


CalibratorInput = Optional[
    Union[str, Path, Mapping[str, Any], SerializedRiskCalibrator]
]


class ReliableOMRRecognizer:
    """Production-sized classical OMR pipeline with review-first routing."""

    def __init__(
        self,
        sheet_definition: DefinitionInput = None,
        calibrator: CalibratorInput = None,
        max_pdf_pages: int = 50,
        include_rectified_image: bool = True,
        preview_max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
        preview_max_height: int = DEFAULT_PREVIEW_MAX_HEIGHT,
        preview_max_jpeg_bytes: int = DEFAULT_PREVIEW_MAX_JPEG_BYTES,
    ):
        self.definition = load_sheet_definition(sheet_definition)
        if isinstance(calibrator, SerializedRiskCalibrator):
            self.calibrator = calibrator
        elif calibrator is None:
            self.calibrator = None
        else:
            self.calibrator = SerializedRiskCalibrator.load(calibrator)
        self.max_pdf_pages = int(max_pdf_pages)
        if self.max_pdf_pages < 1:
            raise ValueError("max_pdf_pages must be positive")
        self.include_rectified_image = bool(include_rectified_image)
        self.preview_max_width = int(preview_max_width)
        self.preview_max_height = int(preview_max_height)
        self.preview_max_jpeg_bytes = int(preview_max_jpeg_bytes)
        if self.include_rectified_image and min(
            self.preview_max_width,
            self.preview_max_height,
            self.preview_max_jpeg_bytes,
        ) <= 0:
            raise ValueError("Rectified preview bounds must be positive")

    def recognize(
        self,
        source: Union[bytes, bytearray, Path, str, np.ndarray],
        capture_mode: str = "scanner",
    ) -> DocumentResult:
        """Recognize all pages in image/PDF bytes, a path, or an ndarray."""

        if capture_mode not in CAPTURE_MODES:
            raise RecognitionInputError(
                "capture_mode must be one of {}".format(sorted(CAPTURE_MODES))
            )
        started = time.perf_counter()
        source_type, images = self._decode_source(source)
        sheets = [
            self.recognize_image(
                image, capture_mode=capture_mode, page_number=index + 1
            )
            for index, image in enumerate(images)
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DocumentResult(
            processor_version=PROCESSOR_VERSION,
            definition_id=self.definition["definition_id"],
            source_type=source_type,
            sheets=sheets,
            processing_ms=round(elapsed_ms, 3),
        )

    def recognize_image(
        self,
        image: np.ndarray,
        capture_mode: str = "scanner",
        page_number: int = 1,
    ) -> SheetResult:
        """Recognize one decoded image and return complete audit diagnostics."""

        if capture_mode not in CAPTURE_MODES:
            raise RecognitionInputError(
                "capture_mode must be one of {}".format(sorted(CAPTURE_MODES))
            )
        quality = assess_input_quality(image, self.definition, capture_mode)
        rectified = rectify_to_canonical(image, self.definition)
        diagnostics = rectified.diagnostics
        rectified_image = (
            encode_rectified_preview(
                rectified.image,
                max_width=self.preview_max_width,
                max_height=self.preview_max_height,
                max_jpeg_bytes=self.preview_max_jpeg_bytes,
            )
            if self.include_rectified_image
            else None
        )
        add_marker_quality(
            quality,
            detected_count=len(
                set(diagnostics.detected_marker_ids)
                & {
                    marker["id"]
                    for marker in self.definition["aruco"]["markers"]
                }
            ),
            required_count=int(self.definition["quality"]["required_markers"]),
            clipped=rectified.markers_clipped,
        )
        max_error = float(
            self.definition["quality"]["max_reprojection_error_px"]
        )
        if (
            diagnostics.reprojection_error_px is not None
            and diagnostics.reprojection_error_px > max_error
        ):
            quality.issues.append(
                QualityIssue(
                    code="marker_reprojection_error",
                    message="Marker homography residual exceeds the configured maximum",
                    severity="error",
                    value=diagnostics.reprojection_error_px,
                    threshold=max_error,
                )
            )
            quality.passed = False

        qr_payload, qr_raw, qr_reasons = self._read_qr(rectified.image)
        questions = recognize_answers(rectified.image, self.definition)
        roll_number, roll_columns = recognize_roll_number(
            rectified.image, self.definition
        )
        features = build_risk_features(
            questions,
            roll_columns,
            quality,
            diagnostics,
            qr_payload,
            self.definition,
        )
        confidence_diagnostics = route_risk(features, self.calibrator)
        risk = confidence_diagnostics.risk
        review_reasons = self._collect_review_reasons(
            questions,
            roll_columns,
            quality,
            diagnostics,
            qr_reasons,
            risk,
        )

        has_fatal = (
            any(issue.severity == "error" for issue in quality.issues)
            or any(
                question.status == MarkStatus.INVALID
                for question in questions + roll_columns
            )
            or any(reason.severity == "error" for reason in qr_reasons)
        )
        has_review_marks = any(
            result.status in {MarkStatus.MULTIPLE, MarkStatus.AMBIGUOUS}
            for result in questions + roll_columns
        )
        if has_fatal:
            status = SheetStatus.INVALID
        elif (
            has_review_marks
            or qr_reasons
            or risk > self.definition["classification"]["accept_risk_max"]
        ):
            status = SheetStatus.REVIEW
        else:
            status = SheetStatus.ACCEPTED

        return SheetResult(
            page_number=int(page_number),
            definition_id=self.definition["definition_id"],
            schema_version=self.definition["schema_version"],
            capture_mode=capture_mode,
            status=status,
            confidence=round(1.0 - risk, 6),
            risk=round(risk, 6),
            quality=quality,
            rectification=diagnostics,
            rectified_image=rectified_image,
            questions=questions,
            roll_number=roll_number,
            roll_columns=roll_columns,
            qr_payload=qr_payload,
            qr_raw=qr_raw,
            confidence_diagnostics=confidence_diagnostics,
            review_reasons=review_reasons,
        )

    def _decode_source(
        self,
        source: Union[bytes, bytearray, Path, str, np.ndarray],
    ) -> Tuple[str, List[np.ndarray]]:
        if isinstance(source, np.ndarray):
            if source.size == 0:
                raise RecognitionInputError("Input image is empty")
            return "array", [source]
        if isinstance(source, (str, Path)):
            path = Path(source)
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise RecognitionInputError(
                    "Could not read input '{}': {}".format(path, exc)
                ) from exc
            force_pdf = path.suffix.lower() == ".pdf"
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            force_pdf = False
        else:
            raise RecognitionInputError(
                "source must be image/PDF bytes, a path, or a NumPy image"
            )
        if not data:
            raise RecognitionInputError("Uploaded file is empty")
        if force_pdf or data[:5] == b"%PDF-":
            return "pdf", self._decode_pdf(data)
        cv2 = require_cv2()
        image = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if image is None:
            raise RecognitionInputError(
                "Input is neither a supported image nor a valid PDF"
            )
        return "image", [image]

    def _decode_pdf(self, data: bytes) -> List[np.ndarray]:
        try:
            import pymupdf as fitz
        except ImportError as exc:  # pragma: no cover - dependency is in requirements
            try:
                import fitz  # type: ignore[no-redef]
            except ImportError:
                raise RecognitionInputError(
                    "PyMuPDF is required to process PDF input"
                ) from exc
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise RecognitionInputError(
                "Could not open PDF: {}".format(exc)
            ) from exc
        try:
            if len(document) == 0:
                raise RecognitionInputError("PDF contains no pages")
            if len(document) > self.max_pdf_pages:
                raise RecognitionInputError(
                    "PDF contains {} pages; maximum is {}".format(
                        len(document), self.max_pdf_pages
                    )
                )
            dpi = float(self.definition["page"]["canonical_dpi"])
            matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            images = []
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=matrix, colorspace=fitz.csGRAY, alpha=False
                )
                images.append(
                    np.frombuffer(pixmap.samples, dtype=np.uint8)
                    .reshape(pixmap.height, pixmap.width)
                    .copy()
                )
            return images
        finally:
            document.close()

    def _read_qr(
        self, canonical_image: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[ReviewReason]]:
        cv2 = require_cv2()
        reasons = []
        try:
            raw, _, _ = cv2.QRCodeDetector().detectAndDecode(canonical_image)
        except Exception:
            raw = ""
        if not raw:
            reasons.append(
                ReviewReason(
                    code="qr_not_detected",
                    message="Expected sheet QR code was not decoded",
                    scope="sheet.qr",
                )
            )
            return None, None, reasons

        try:
            payload = parse_qr_payload(raw, self.definition)
        except QRPayloadError as exc:
            reasons.append(
                ReviewReason(
                    code=exc.code,
                    message=str(exc),
                    scope=exc.scope,
                    severity="error",
                )
            )
            return None, raw, reasons
        return payload, raw, reasons

    def _collect_review_reasons(
        self,
        questions: List[Any],
        roll_columns: List[Any],
        quality: Any,
        diagnostics: Any,
        qr_reasons: List[ReviewReason],
        risk: float,
    ) -> List[ReviewReason]:
        reasons = [
            ReviewReason(
                code=issue.code,
                message=issue.message,
                scope="sheet.quality",
                severity=issue.severity,
            )
            for issue in quality.issues
        ]
        if diagnostics.method != "aruco_homography":
            reasons.append(
                ReviewReason(
                    code="rectification_fallback",
                    message=diagnostics.fallback_reason
                    or "ArUco homography was not available",
                    scope="sheet.rectification",
                    severity="error",
                )
            )
        for result in questions:
            reasons.extend(result.review_reasons)
        for result in roll_columns:
            reasons.extend(result.review_reasons)
        reasons.extend(qr_reasons)
        if risk > self.definition["classification"]["accept_risk_max"]:
            reasons.append(
                ReviewReason(
                    code="risk_above_acceptance_threshold",
                    message="Estimated risk exceeds the auto-accept threshold",
                    scope="sheet.confidence",
                )
            )
        return reasons
