"""Validated exports from human labels and Reliable OMR diagnostics."""

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from src.reliable_omr.calibration import RISK_FEATURES
from src.reliable_omr.modeling.contracts import (
    BUBBLE_EXPORT_FIELDS,
    BUBBLE_LABELS,
    BUBBLE_DATASET_SCHEMA_VERSION,
    CAPTURE_MODES,
    CAPTURE_REQUIRED_FIELDS,
    DATASET_SCHEMA_VERSION,
    HUMAN_ANNOTATION_REQUIRED_FIELDS,
)
from src.reliable_omr.modeling.dataset import (
    DatasetContractError,
    load_capture_rows,
    sha256_file,
)
from src.reliable_omr.opencv import require_cv2


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MACHINE_LABELS = {"empty", "filled", "ambiguous", "invalid"}
CAPTURE_METADATA_SCHEMA_VERSION = "omr-capture-metadata-v1"
CAPTURE_VERIFICATION_SCHEMA_VERSION = "omr-capture-verification-v1"


class LabelingContractError(DatasetContractError):
    """Raised before an invalid labeling export can be written."""


def _load_json_object(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LabelingContractError(
            "Unable to read {} JSON '{}': {}".format(description, path, exc)
        ) from exc
    if not isinstance(value, dict):
        raise LabelingContractError(
            "{} JSON must contain an object".format(description)
        )
    return value


def _identifier(value: Any, field: str) -> str:
    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise LabelingContractError(
            "{} must be a stable identifier using letters, numbers, '.', "
            "'_', ':', or '-'".format(field)
        )
    return normalized


def _metadata(path: Path) -> Dict[str, str]:
    payload = _load_json_object(path, "capture metadata")
    if payload.get("schema_version") != CAPTURE_METADATA_SCHEMA_VERSION:
        raise LabelingContractError(
            "Capture metadata schema_version must be '{}'".format(
                CAPTURE_METADATA_SCHEMA_VERSION
            )
        )
    output = {
        field: _identifier(payload.get(field), field)
        for field in (
            "capture_id",
            "physical_sheet_id",
            "capture_session_id",
            "device_id",
            "print_batch_id",
        )
    }
    capture_mode = str(payload.get("capture_mode", "")).strip().lower()
    if capture_mode not in CAPTURE_MODES:
        raise LabelingContractError(
            "capture_mode must be one of {}".format(list(CAPTURE_MODES))
        )
    output["capture_mode"] = capture_mode
    return output


def _sheet_payload(
    processor_payload: Mapping[str, Any], sheet_index: int
) -> Mapping[str, Any]:
    if "sheets" not in processor_payload:
        if sheet_index != 0:
            raise LabelingContractError(
                "sheet_index must be zero for a single-sheet result"
            )
        return processor_payload
    sheets = processor_payload.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise LabelingContractError(
            "Processor result 'sheets' must be a non-empty list"
        )
    if sheet_index < 0 or sheet_index >= len(sheets):
        raise LabelingContractError("sheet_index is outside processor results")
    sheet = sheets[sheet_index]
    if not isinstance(sheet, dict):
        raise LabelingContractError(
            "Selected processor sheet must be an object"
        )
    return sheet


def _finite_float(value: Any, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelingContractError(
            "{} must be numeric".format(field)
        ) from exc
    if not math.isfinite(numeric):
        raise LabelingContractError("{} must be finite".format(field))
    return numeric


def _validated_risk_feature(value: Any, field: str) -> float:
    numeric = _finite_float(value, field)
    unit_interval = {
        "rectification_fallback",
        "ambiguous_fraction",
        "multiple_fraction",
        "invalid_fraction",
        "low_margin_fraction",
        "mean_question_confidence",
        "min_question_confidence",
        "roll_uncertain_fraction",
        "qr_missing",
    }
    non_negative = {
        "quality_error_count",
        "quality_review_count",
        "reprojection_error_norm",
    }
    if field in unit_interval and not 0.0 <= numeric <= 1.0:
        raise LabelingContractError("{} must be in [0, 1]".format(field))
    if field in non_negative and numeric < 0.0:
        raise LabelingContractError("{} must be non-negative".format(field))
    if field in {"rectification_fallback", "qr_missing"} and numeric not in {
        0.0,
        1.0,
    }:
        raise LabelingContractError("{} must be binary".format(field))
    return numeric


def _bounded_box(
    box: Mapping[str, Any], image_width: int, image_height: int, scope: str
) -> Tuple[int, int, int, int]:
    try:
        values = [box[field] for field in ("x", "y", "width", "height")]
    except KeyError as exc:
        raise LabelingContractError(
            "{} bounding_box must have integer x/y/width/height".format(scope)
        ) from exc
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in values
    ):
        raise LabelingContractError(
            "{} bounding_box must have integer x/y/width/height".format(scope)
        )
    x, y, width, height = values
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise LabelingContractError(
            "{} bounding_box extends outside the rectified image".format(scope)
        )
    return x, y, width, height


def _processor_bubbles(
    sheet: Mapping[str, Any],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    questions = sheet.get("questions")
    if not isinstance(questions, list) or not questions:
        raise LabelingContractError(
            "Processor result must contain non-empty questions"
        )
    output: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for question_index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise LabelingContractError(
                "Processor question {} must be an object".format(
                    question_index
                )
            )
        question_key = _identifier(
            question.get("field_id"), "processor question field_id"
        )
        bounding_box = question.get("bounding_box")
        if not isinstance(bounding_box, dict):
            raise LabelingContractError(
                "{} is missing bounding_box".format(question_key)
            )
        _bounded_box(
            bounding_box, image_width, image_height, question_key
        )
        bubbles = question.get("bubbles")
        if not isinstance(bubbles, list) or not bubbles:
            raise LabelingContractError(
                "{} must contain bubbles".format(question_key)
            )
        for bubble in bubbles:
            if not isinstance(bubble, dict):
                raise LabelingContractError(
                    "{} contains a non-object bubble".format(question_key)
                )
            option_key = _identifier(
                bubble.get("option"), "{} option".format(question_key)
            )
            key = (question_key, option_key)
            if key in output:
                raise LabelingContractError(
                    "Duplicate processor bubble {} / {}".format(*key)
                )
            center = bubble.get("center_px")
            if not isinstance(center, list) or len(center) != 2:
                raise LabelingContractError(
                    "{} / {} center_px must contain x/y".format(*key)
                )
            center_x = _finite_float(center[0], "bubble center x")
            center_y = _finite_float(center[1], "bubble center y")
            radius = _finite_float(bubble.get("radius_px"), "bubble radius")
            if radius <= 0:
                raise LabelingContractError("bubble radius must be positive")
            half_extent = max(1.0, radius * padding_ratio)
            left = int(math.floor(center_x - half_extent))
            top = int(math.floor(center_y - half_extent))
            right = int(math.ceil(center_x + half_extent))
            bottom = int(math.ceil(center_y + half_extent))
            bbox = {
                "x": left,
                "y": top,
                "width": right - left,
                "height": bottom - top,
            }
            _bounded_box(
                bbox,
                image_width,
                image_height,
                "{} / {}".format(*key),
            )
            machine_label = str(bubble.get("status", "")).strip().lower()
            if machine_label not in _MACHINE_LABELS:
                raise LabelingContractError(
                    "{} / {} has invalid machine bubble status".format(*key)
                )
            machine_confidence = _finite_float(
                bubble.get("confidence"), "machine bubble confidence"
            )
            if not 0.0 <= machine_confidence <= 1.0:
                raise LabelingContractError(
                    "machine bubble confidence must be in [0, 1]"
                )
            output[key] = {
                "bbox": bbox,
                "machine_label": machine_label,
                "machine_confidence": machine_confidence,
            }
    return output


def _annotation_rows(
    path: Path, capture_id: str
) -> Dict[Tuple[str, str], Dict[str, str]]:
    try:
        with Path(path).open(
            "r", encoding="utf-8-sig", newline=""
        ) as source:
            reader = csv.DictReader(source)
            fields = list(reader.fieldnames or [])
            if not fields:
                raise LabelingContractError(
                    "Human annotations CSV has no header"
                )
            if len(fields) != len(set(fields)):
                raise LabelingContractError(
                    "Human annotations CSV has duplicate columns"
                )
            missing = sorted(
                set(HUMAN_ANNOTATION_REQUIRED_FIELDS) - set(fields)
            )
            if missing:
                raise LabelingContractError(
                    "Human annotations CSV is missing columns: {}".format(
                        missing
                    )
                )
            raw_rows = list(reader)
    except OSError as exc:
        raise LabelingContractError(
            "Unable to read human annotations CSV: {}".format(exc)
        ) from exc
    if not raw_rows:
        raise LabelingContractError("Human annotations CSV has no records")

    output = {}
    for row_number, row in enumerate(raw_rows, start=2):
        if None in row:
            raise LabelingContractError(
                "Human annotation row {} has too many values".format(
                    row_number
                )
            )
        if _identifier(row.get("capture_id"), "capture_id") != capture_id:
            raise LabelingContractError(
                "Human annotation row {} capture_id does not match "
                "metadata".format(
                    row_number
                )
            )
        question_key = _identifier(
            row.get("question_key"), "question_key"
        )
        option_key = _identifier(row.get("option_key"), "option_key")
        key = (question_key, option_key)
        if key in output:
            raise LabelingContractError(
                "Duplicate human annotation {} / {}".format(*key)
            )
        reviewer_1 = _identifier(
            row.get("reviewer_1_id"), "reviewer_1_id"
        )
        reviewer_2 = _identifier(
            row.get("reviewer_2_id"), "reviewer_2_id"
        )
        if reviewer_1 == reviewer_2:
            raise LabelingContractError(
                "Human annotation reviewers must be different people"
            )
        reviewer_1_label = str(row.get("reviewer_1_label", "")).strip().lower()
        reviewer_2_label = str(row.get("reviewer_2_label", "")).strip().lower()
        final_label = str(row.get("label", "")).strip().lower()
        for label in (reviewer_1_label, reviewer_2_label, final_label):
            if label not in BUBBLE_LABELS:
                raise LabelingContractError(
                    "Human annotation labels must be one of {}".format(
                        list(BUBBLE_LABELS)
                    )
                )
        status = str(row.get("label_status", "")).strip().lower()
        adjudicator = str(row.get("adjudicator_id", "")).strip()
        if reviewer_1_label == reviewer_2_label:
            if (
                final_label != reviewer_1_label
                or status != "verified"
                or adjudicator
            ):
                raise LabelingContractError(
                    "Agreeing reviewers require their shared label, verified "
                    "status, and no adjudicator"
                )
        else:
            adjudicator = _identifier(adjudicator, "adjudicator_id")
            if (
                status != "adjudicated"
                or adjudicator in {reviewer_1, reviewer_2}
            ):
                raise LabelingContractError(
                    "Disagreeing reviewers require an independent adjudicator "
                    "and adjudicated status"
                )
        output[key] = {"label": final_label, "label_status": status}
    return output


def _safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return "{}-{}".format(stem, suffix)


def _relative_path(path: Path, relative_to: Path) -> str:
    return Path(os.path.relpath(path, relative_to)).as_posix()


def export_bubble_crops(
    rectified_image_path: Path,
    processor_result_path: Path,
    human_annotations_path: Path,
    capture_metadata_path: Path,
    output_dir: Path,
    sheet_index: int = 0,
    crop_size: int = 64,
    padding_ratio: float = 1.65,
) -> Dict[str, Any]:
    """Write complete human-labelled canonical bubble crops and a v1 CSV."""

    if crop_size < 16 or crop_size > 512:
        raise LabelingContractError("crop_size must be between 16 and 512")
    if not 1.0 <= padding_ratio <= 4.0:
        raise LabelingContractError("padding_ratio must be between 1 and 4")

    rectified_image_path = Path(rectified_image_path)
    processor_result_path = Path(processor_result_path)
    human_annotations_path = Path(human_annotations_path)
    capture_metadata_path = Path(capture_metadata_path)
    metadata = _metadata(capture_metadata_path)
    processor_payload = _load_json_object(
        processor_result_path, "processor result"
    )
    sheet = _sheet_payload(processor_payload, sheet_index)
    sheet_mode = str(sheet.get("capture_mode", "")).strip().lower()
    if sheet_mode and sheet_mode != metadata["capture_mode"]:
        raise LabelingContractError(
            "Processor capture_mode does not match capture metadata"
        )

    cv2 = require_cv2()
    image = cv2.imread(str(rectified_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise LabelingContractError(
            "Rectified image could not be decoded as grayscale"
        )
    height, width = image.shape
    bubbles = _processor_bubbles(
        sheet, width, height, float(padding_ratio)
    )
    annotations = _annotation_rows(
        human_annotations_path, metadata["capture_id"]
    )
    missing_annotations = sorted(set(bubbles) - set(annotations))
    unknown_annotations = sorted(set(annotations) - set(bubbles))
    if missing_annotations or unknown_annotations:
        raise LabelingContractError(
            "Human/processor bubble keys differ; missing={} unknown={}".format(
                missing_annotations[:5], unknown_annotations[:5]
            )
        )

    provenance = {
        "rectified_image_sha256": sha256_file(rectified_image_path),
        "processor_result_sha256": sha256_file(processor_result_path),
        "human_annotations_sha256": sha256_file(human_annotations_path),
        "capture_metadata_sha256": sha256_file(capture_metadata_path),
    }
    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    output_csv = output_dir / "bubble_crops.csv"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for question_key, option_key in sorted(bubbles):
        bubble = bubbles[(question_key, option_key)]
        annotation = annotations[(question_key, option_key)]
        bbox = bubble["bbox"]
        crop = image[
            bbox["y"] : bbox["y"] + bbox["height"],
            bbox["x"] : bbox["x"] + bbox["width"],
        ]
        interpolation = (
            cv2.INTER_AREA
            if crop.shape[0] > crop_size or crop.shape[1] > crop_size
            else cv2.INTER_LINEAR
        )
        normalized = cv2.resize(
            crop, (crop_size, crop_size), interpolation=interpolation
        )
        crop_id = "{}.{}.{}".format(
            metadata["capture_id"], question_key, option_key
        )
        crop_id = _identifier(crop_id, "crop_id")
        crop_path = crops_dir / "{}.png".format(_safe_filename(crop_id))
        if not cv2.imwrite(str(crop_path), normalized):
            raise LabelingContractError(
                "Failed to write crop '{}'".format(crop_path)
            )
        row = {
            "schema_version": BUBBLE_DATASET_SCHEMA_VERSION,
            "crop_id": crop_id,
            **metadata,
            "question_key": question_key,
            "option_key": option_key,
            "crop_path": _relative_path(crop_path, output_csv.parent),
            "crop_sha256": sha256_file(crop_path),
            "label": annotation["label"],
            "label_status": annotation["label_status"],
            "bbox_x": bbox["x"],
            "bbox_y": bbox["y"],
            "bbox_width": bbox["width"],
            "bbox_height": bbox["height"],
            **provenance,
            "machine_label": bubble["machine_label"],
            "machine_confidence": format(
                float(bubble["machine_confidence"]), ".17g"
            ),
        }
        rows.append(row)

    with output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(BUBBLE_EXPORT_FIELDS),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "bubble_csv": output_csv,
        "crops_dir": crops_dir,
        "crop_count": len(rows),
        "provenance": provenance,
    }


def _binary_value(value: Any, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip()
    if normalized not in {"0", "1"}:
        raise LabelingContractError("{} must be exactly 0 or 1".format(field))
    return int(normalized)


def _capture_verification(path: Path) -> Dict[str, Any]:
    payload = _load_json_object(path, "capture verification")
    if payload.get("schema_version") != CAPTURE_VERIFICATION_SCHEMA_VERSION:
        raise LabelingContractError(
            "Capture verification schema_version must be '{}'".format(
                CAPTURE_VERIFICATION_SCHEMA_VERSION
            )
        )
    reviewer_1 = _identifier(payload.get("reviewer_1_id"), "reviewer_1_id")
    reviewer_2 = _identifier(payload.get("reviewer_2_id"), "reviewer_2_id")
    if reviewer_1 == reviewer_2:
        raise LabelingContractError(
            "Capture verification reviewers must be different people"
        )
    label_1 = _binary_value(
        payload.get("reviewer_1_is_error"), "reviewer_1_is_error"
    )
    label_2 = _binary_value(
        payload.get("reviewer_2_is_error"), "reviewer_2_is_error"
    )
    final_label = _binary_value(payload.get("is_error"), "is_error")
    status = str(payload.get("label_status", "")).strip().lower()
    adjudicator = str(payload.get("adjudicator_id", "")).strip()
    if label_1 == label_2:
        if final_label != label_1 or status != "verified" or adjudicator:
            raise LabelingContractError(
                "Agreeing capture reviewers require their shared label, "
                "verified status, and no adjudicator"
            )
    else:
        adjudicator = _identifier(adjudicator, "adjudicator_id")
        if (
            status != "adjudicated"
            or adjudicator in {reviewer_1, reviewer_2}
        ):
            raise LabelingContractError(
                "Disagreeing capture reviewers require an independent "
                "adjudicator and adjudicated status"
            )
    return {"is_error": final_label, "label_status": status}


def export_capture_row(
    capture_image_path: Path,
    processor_result_path: Path,
    capture_metadata_path: Path,
    human_verification_path: Path,
    output_csv: Path,
    sheet_index: int = 0,
) -> Dict[str, Any]:
    """Export one capture row; machine output never supplies human truth."""

    capture_image_path = Path(capture_image_path)
    processor_result_path = Path(processor_result_path)
    capture_metadata_path = Path(capture_metadata_path)
    human_verification_path = Path(human_verification_path)
    output_csv = Path(output_csv)
    if not capture_image_path.is_file():
        raise LabelingContractError("Capture image does not exist")
    metadata = _metadata(capture_metadata_path)
    verification = _capture_verification(human_verification_path)
    processor_payload = _load_json_object(
        processor_result_path, "processor result"
    )
    sheet = _sheet_payload(processor_payload, sheet_index)
    diagnostics = sheet.get("confidence_diagnostics")
    if not isinstance(diagnostics, dict):
        raise LabelingContractError(
            "Processor sheet is missing confidence_diagnostics"
        )
    features = diagnostics.get("features")
    if not isinstance(features, dict):
        raise LabelingContractError(
            "confidence_diagnostics.features must be an object"
        )
    missing_features = sorted(set(RISK_FEATURES) - set(features))
    unexpected_features = sorted(set(features) - set(RISK_FEATURES))
    if missing_features or unexpected_features:
        raise LabelingContractError(
            "confidence_diagnostics feature keys differ; missing={} "
            "unexpected={}".format(missing_features, unexpected_features)
        )
    sheet_mode = str(sheet.get("capture_mode", "")).strip().lower()
    if sheet_mode and sheet_mode != metadata["capture_mode"]:
        raise LabelingContractError(
            "Processor capture_mode does not match capture metadata"
        )

    row = {
        "schema_version": DATASET_SCHEMA_VERSION,
        **metadata,
        "image_path": _relative_path(capture_image_path, output_csv.parent),
        "capture_sha256": sha256_file(capture_image_path),
        **verification,
        "processor_result_sha256": sha256_file(processor_result_path),
        "capture_metadata_sha256": sha256_file(capture_metadata_path),
        "human_verification_sha256": sha256_file(human_verification_path),
    }
    for feature in RISK_FEATURES:
        row[feature] = format(
            _validated_risk_feature(features[feature], feature), ".17g"
        )
    fieldnames = [
        *CAPTURE_REQUIRED_FIELDS,
        "capture_sha256",
        *RISK_FEATURES,
        "processor_result_sha256",
        "capture_metadata_sha256",
        "human_verification_sha256",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(row)
    load_capture_rows(output_csv)
    return {
        "capture_csv": output_csv,
        "capture_id": metadata["capture_id"],
        "provenance": {
            key: row[key]
            for key in (
                "capture_sha256",
                "processor_result_sha256",
                "capture_metadata_sha256",
                "human_verification_sha256",
            )
        },
    }
