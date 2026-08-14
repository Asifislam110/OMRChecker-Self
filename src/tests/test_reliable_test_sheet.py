import pytest

cv2 = pytest.importorskip("cv2")

from src.reliable_omr.recognizer import ReliableOMRRecognizer  # noqa: E402
from src.reliable_omr.test_sheet import (  # noqa: E402
    compact_qr_payload,
    cycling_answers,
    generate_canonical_test_sheet,
)
from src.reliable_omr.types import SheetStatus  # noqa: E402


def test_canonical_diagnostic_sheet_round_trips_all_contract_features():
    if not hasattr(cv2, "aruco"):
        pytest.skip("opencv-contrib ArUco is not installed")
    answers = cycling_answers()
    image = generate_canonical_test_sheet(answers=answers)

    result = ReliableOMRRecognizer(
        include_rectified_image=False
    ).recognize_image(image)

    assert image.shape == (3508, 2480, 3)
    assert result.status == SheetStatus.ACCEPTED
    assert result.rectification.detected_marker_ids == [0, 1, 2, 3]
    assert result.qr_raw == "O1:test.1.deadbeef"
    assert result.qr_payload == {
        "exam_id": "test",
        "template_version": "1",
        "template_checksum": "deadbeef",
    }
    assert result.roll_number == "12345678"
    assert len(result.questions) == 100
    assert all(answers[row.question] == row.answer for row in result.questions)


def test_compact_qr_rejects_payloads_beyond_verified_capacity():
    with pytest.raises(ValueError, match="capacity"):
        compact_qr_payload("x" * 30, "1", "deadbeef")
