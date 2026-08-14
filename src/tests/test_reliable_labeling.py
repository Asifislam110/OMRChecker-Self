import csv
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.reliable_omr.calibration import RISK_FEATURES  # noqa: E402
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    load_capture_rows,
    sha256_file,
)
from src.reliable_omr.modeling.labeling import (  # noqa: E402
    LabelingContractError,
    export_bubble_crops,
    export_capture_row,
)


def _write_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = np.full((200, 200), 245, dtype=np.uint8)
    cv2.circle(image, (60, 70), 12, 20, 2)
    cv2.circle(image, (120, 130), 12, 20, 2)
    cv2.line(image, (110, 130), (130, 130), 30, 4)
    image_path = tmp_path / "rectified.png"
    assert cv2.imwrite(str(image_path), image)

    features = {name: 0.0 for name in RISK_FEATURES}
    features.update(
        {
            "mean_question_confidence": 0.91,
            "min_question_confidence": 0.72,
        }
    )
    processor = {
        "capture_mode": "scanner",
        "status": "accepted",
        "questions": [
            {
                "field_id": "q1",
                "bounding_box": {
                    "x": 40,
                    "y": 50,
                    "width": 40,
                    "height": 40,
                },
                "bubbles": [
                    {
                        "option": "A",
                        "center_px": [60.0, 70.0],
                        "radius_px": 12.0,
                        "status": "filled",
                        "confidence": 0.99,
                    }
                ],
            },
            {
                "field_id": "q2",
                "bounding_box": {
                    "x": 100,
                    "y": 110,
                    "width": 40,
                    "height": 40,
                },
                "bubbles": [
                    {
                        "option": "B",
                        "center_px": [120.0, 130.0],
                        "radius_px": 12.0,
                        "status": "ambiguous",
                        "confidence": 0.51,
                    }
                ],
            },
        ],
        "confidence_diagnostics": {"features": features},
    }
    processor_path = tmp_path / "processor.json"
    processor_path.write_text(json.dumps(processor), encoding="utf-8")

    metadata = {
        "schema_version": "omr-capture-metadata-v1",
        "capture_id": "capture-1",
        "physical_sheet_id": "sheet-1",
        "capture_session_id": "session-1",
        "device_id": "scanner-1",
        "print_batch_id": "batch-1",
        "capture_mode": "scanner",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    annotations_path = tmp_path / "annotations.csv"
    fields = [
        "capture_id",
        "question_key",
        "option_key",
        "reviewer_1_id",
        "reviewer_1_label",
        "reviewer_2_id",
        "reviewer_2_label",
        "adjudicator_id",
        "label",
        "label_status",
    ]
    with annotations_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(
            {
                "capture_id": "capture-1",
                "question_key": "q1",
                "option_key": "A",
                "reviewer_1_id": "reviewer-a",
                "reviewer_1_label": "empty",
                "reviewer_2_id": "reviewer-b",
                "reviewer_2_label": "empty",
                "adjudicator_id": "",
                "label": "empty",
                "label_status": "verified",
            }
        )
        writer.writerow(
            {
                "capture_id": "capture-1",
                "question_key": "q2",
                "option_key": "B",
                "reviewer_1_id": "reviewer-a",
                "reviewer_1_label": "partial",
                "reviewer_2_id": "reviewer-b",
                "reviewer_2_label": "crossed",
                "adjudicator_id": "reviewer-c",
                "label": "partial",
                "label_status": "adjudicated",
            }
        )
    return image_path, processor_path, metadata_path, annotations_path


def test_bubble_export_uses_human_truth_and_hashes_every_source(tmp_path):
    inputs = _write_inputs(tmp_path)
    image_path, processor_path, metadata_path, annotations_path = inputs

    result = export_bubble_crops(
        image_path,
        processor_path,
        annotations_path,
        metadata_path,
        tmp_path / "export",
    )
    rows = list(
        csv.DictReader(result["bubble_csv"].open(encoding="utf-8"))
    )

    assert result["crop_count"] == 2
    assert rows[0]["label"] == "empty"
    assert rows[0]["machine_label"] == "filled"
    assert rows[0]["capture_session_id"] == "session-1"
    assert rows[0]["device_id"] == "scanner-1"
    assert rows[0]["print_batch_id"] == "batch-1"
    assert rows[0]["rectified_image_sha256"] == sha256_file(image_path)
    assert rows[0]["processor_result_sha256"] == sha256_file(processor_path)
    assert rows[0]["human_annotations_sha256"] == sha256_file(
        annotations_path
    )
    crop_path = result["bubble_csv"].parent / rows[0]["crop_path"]
    assert rows[0]["crop_sha256"] == sha256_file(crop_path)
    assert cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE).shape == (64, 64)


def test_bubble_export_rejects_unknown_keys_and_out_of_bounds_geometry(
    tmp_path,
):
    inputs = _write_inputs(tmp_path)
    image_path, processor_path, metadata_path, annotations_path = inputs
    rows = list(csv.DictReader(annotations_path.open(encoding="utf-8")))
    rows[0]["option_key"] = "Z"
    with annotations_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(LabelingContractError, match="bubble keys differ"):
        export_bubble_crops(
            image_path,
            processor_path,
            annotations_path,
            metadata_path,
            tmp_path / "bad-keys",
        )

    _, processor_path, _, annotations_path = _write_inputs(
        tmp_path / "bounds"
    )
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    processor["questions"][0]["bubbles"][0]["center_px"] = [2.0, 2.0]
    processor_path.write_text(json.dumps(processor), encoding="utf-8")
    with pytest.raises(LabelingContractError, match="outside"):
        export_bubble_crops(
            tmp_path / "bounds" / "rectified.png",
            processor_path,
            annotations_path,
            tmp_path / "bounds" / "metadata.json",
            tmp_path / "bad-bounds",
        )


def test_capture_export_uses_explicit_two_reviewer_error_label(tmp_path):
    image_path, processor_path, metadata_path, _ = _write_inputs(tmp_path)
    verification = {
        "schema_version": "omr-capture-verification-v1",
        "reviewer_1_id": "reviewer-a",
        "reviewer_1_is_error": 0,
        "reviewer_2_id": "reviewer-b",
        "reviewer_2_is_error": 1,
        "adjudicator_id": "reviewer-c",
        "is_error": 1,
        "label_status": "adjudicated",
    }
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(verification), encoding="utf-8")

    result = export_capture_row(
        image_path,
        processor_path,
        metadata_path,
        verification_path,
        tmp_path / "capture.csv",
    )
    rows, extra = load_capture_rows(result["capture_csv"])

    assert rows[0]["is_error"] == 1
    assert rows[0]["label_status"] == "adjudicated"
    assert rows[0]["mean_question_confidence"] == pytest.approx(0.91)
    assert "processor_result_sha256" in extra
    assert result["provenance"]["human_verification_sha256"] == sha256_file(
        verification_path
    )
