import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.reliable_omr.definition import (  # noqa: E402
    iter_answer_geometry,
    iter_roll_geometry,
    load_sheet_definition,
    mm_to_px,
    point_mm_to_px,
)
from src.reliable_omr.recognizer import ReliableOMRRecognizer  # noqa: E402
from src.reliable_omr.types import MarkStatus, SheetStatus  # noqa: E402


def _marker_image(dictionary, marker_id, size):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, size, image, 1)
    return image


def synthetic_sheet():
    if not hasattr(cv2, "aruco"):
        pytest.skip("opencv-contrib ArUco is not installed")
    definition = load_sheet_definition()
    page = definition["page"]
    image = np.full(
        (page["canonical_height_px"], page["canonical_width_px"]),
        255,
        dtype=np.uint8,
    )
    dictionary_id = getattr(
        cv2.aruco, definition["aruco"]["dictionary"]
    )
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    for marker in definition["aruco"]["markers"]:
        x, y = [
            int(round(value))
            for value in point_mm_to_px(marker["top_left_mm"], definition)
        ]
        size = int(round(mm_to_px(marker["size_mm"], definition)))
        image[y : y + size, x : x + size] = _marker_image(
            dictionary, marker["id"], size
        )

    for bubble in iter_answer_geometry(definition):
        center = tuple(
            int(round(value))
            for value in point_mm_to_px(bubble["center_mm"], definition)
        )
        radius = int(round(mm_to_px(bubble["diameter_mm"], definition) / 2))
        cv2.circle(image, center, radius, 0, 3)
        if bubble["option"] == "A":
            cv2.circle(image, center, max(2, int(radius * 0.55)), 25, -1)

    roll_digits = "12345678"
    for bubble in iter_roll_geometry(definition):
        center = tuple(
            int(round(value))
            for value in point_mm_to_px(bubble["center_mm"], definition)
        )
        radius = int(round(mm_to_px(bubble["diameter_mm"], definition) / 2))
        cv2.circle(image, center, radius, 0, 2)
        if str(bubble["digit"]) == roll_digits[bubble["column_index"]]:
            cv2.circle(image, center, max(2, int(radius * 0.55)), 25, -1)
    return image


def test_synthetic_golden_sheet_rectifies_and_reads_fixed_geometry():
    image = synthetic_sheet()
    result = ReliableOMRRecognizer().recognize_image(image)

    assert result.rectification.method == "aruco_homography"
    assert result.rectification.detected_marker_ids == [0, 1, 2, 3]
    assert result.questions[0].status == MarkStatus.FILLED
    assert result.questions[0].answer == "A"
    assert {question.answer for question in result.questions} == {"A"}
    assert len(result.questions) == 100
    assert result.questions[0].bounding_box.x > 0
    assert result.questions[0].bounding_box.width > 0
    assert result.questions[0].bounding_box.height > 0
    first_box = result.questions[0].bounding_box
    for bubble in result.questions[0].bubbles:
        assert first_box.x <= bubble.center_px[0] - bubble.radius_px
        assert first_box.y <= bubble.center_px[1] - bubble.radius_px
        assert (
            first_box.x + first_box.width
            >= bubble.center_px[0] + bubble.radius_px
        )
        assert (
            first_box.y + first_box.height
            >= bubble.center_px[1] + bubble.radius_px
        )
    assert result.roll_number == "12345678"
    assert result.rectified_image.content_type == "image/jpeg"
    assert result.rectified_image.width <= 1000
    assert result.rectified_image.height <= 1415
    assert result.status == SheetStatus.REVIEW
    assert any(reason.code == "qr_not_detected" for reason in result.review_reasons)


def test_missing_marker_returns_structured_invalid_fallback():
    image = synthetic_sheet()
    definition = load_sheet_definition()
    marker = definition["aruco"]["markers"][0]
    x, y = [
        int(round(value))
        for value in point_mm_to_px(marker["top_left_mm"], definition)
    ]
    size = int(round(mm_to_px(marker["size_mm"], definition)))
    image[y : y + size, x : x + size] = 255

    result = ReliableOMRRecognizer().recognize_image(image)

    assert result.status == SheetStatus.INVALID
    assert result.rectification.method == "aspect_resize_fallback"
    assert result.rectification.missing_marker_ids == [0]
    assert any(
        issue.code == "markers_incomplete" for issue in result.quality.issues
    )


def test_aruco_unavailable_returns_diagnostics_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(
        "src.reliable_omr.rectification.has_aruco", lambda: False
    )

    result = ReliableOMRRecognizer(
        include_rectified_image=False
    ).recognize_image(synthetic_sheet())

    assert result.status == SheetStatus.INVALID
    assert result.rectified_image is None
    assert result.rectification.aruco_available is False
    assert result.rectification.missing_marker_ids == [0, 1, 2, 3]
    assert "unavailable" in result.rectification.fallback_reason


def test_pdf_bytes_are_rendered_as_individual_sheets():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page(width=595, height=842)
    pdf_bytes = document.tobytes()
    document.close()

    result = ReliableOMRRecognizer().recognize(
        pdf_bytes, capture_mode="scanner"
    )

    assert result.source_type == "pdf"
    assert len(result.sheets) == 1
    assert result.sheets[0].page_number == 1
    assert result.sheets[0].status == SheetStatus.INVALID
    assert result.processor_version == "reliable-omr/1.1.0"


def test_compact_product_qr_payload_is_supported(monkeypatch):
    class Detector:
        def detectAndDecode(self, _image):
            return "O1:abc.2.deadbeef", None, None

    monkeypatch.setattr(cv2, "QRCodeDetector", lambda: Detector())
    payload, raw, reasons = ReliableOMRRecognizer()._read_qr(
        np.full((100, 100), 255, dtype=np.uint8)
    )

    assert raw == "O1:abc.2.deadbeef"
    assert payload == {
        "exam_id": "abc",
        "template_version": "2",
        "template_checksum": "deadbeef",
    }
    assert reasons == []
