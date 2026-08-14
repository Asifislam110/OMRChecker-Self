import csv

import pytest

from src.reliable_omr.calibration import RISK_FEATURES
from src.reliable_omr.modeling.contracts import (
    BUBBLE_DATASET_SCHEMA_VERSION,
    BUBBLE_EXPORT_FIELDS,
    CAPTURE_REQUIRED_FIELDS,
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    GROUP_FIELDS,
)
from src.reliable_omr.modeling.dataset import (
    DatasetContractError,
    load_capture_rows,
    load_prepared_splits,
    prepare_capture_dataset,
    sha256_file,
)
from src.reliable_omr.modeling.readiness import classifier_readiness_report
from src.reliable_omr.training import train_serialized_calibrator


def _feature_values(is_error):
    values = {name: 0.0 for name in RISK_FEATURES}
    values.update(
        {
            "quality_error_count": float(is_error),
            "ambiguous_fraction": 0.6 if is_error else 0.02,
            "mean_question_confidence": 0.35 if is_error else 0.95,
            "min_question_confidence": 0.2 if is_error else 0.85,
        }
    )
    return values


def _write_capture_csv(tmp_path, duplicate_content=False):
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for group_index in range(30):
        mode = "scanner" if group_index % 2 == 0 else "mobile"
        is_error = group_index % 2
        for repeat in range(2):
            capture_id = "capture-{:02d}-{}".format(group_index, repeat)
            image_path = image_dir / "{}.jpg".format(capture_id)
            content_id = (
                "duplicate"
                if duplicate_content and group_index == 0
                else capture_id
            )
            image_path.write_bytes(
                "real-placeholder-bytes-{}".format(content_id).encode("ascii")
            )
            row = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "capture_id": capture_id,
                "image_path": "images/{}.jpg".format(capture_id),
                "physical_sheet_id": "sheet-{:02d}".format(group_index),
                "capture_session_id": "session-{:02d}".format(group_index),
                "device_id": "device-{:02d}".format(group_index),
                "print_batch_id": "batch-{:02d}".format(group_index),
                "capture_mode": mode,
                "is_error": is_error,
                "label_status": "verified",
                "pen_type": "pen" if group_index % 3 else "pencil",
            }
            row.update(_feature_values(is_error))
            rows.append(row)

    csv_path = tmp_path / "captures.csv"
    fields = [*CAPTURE_REQUIRED_FIELDS, *RISK_FEATURES, "pen_type"]
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_preparation_is_deterministic_group_safe_and_calibrator_compatible(
    tmp_path,
):
    source = _write_capture_csv(tmp_path)
    first = prepare_capture_dataset(source, tmp_path / "prepared-a", seed=9)
    second = prepare_capture_dataset(source, tmp_path / "prepared-b", seed=9)

    assert first["normalized_csv"].read_bytes() == second[
        "normalized_csv"
    ].read_bytes()
    assert first["split_manifest"].read_bytes() == second[
        "split_manifest"
    ].read_bytes()
    assert first["audit"]["overall"]["capture_count"] == 60
    assert first["manifest"]["training_readiness"]["ready"] is True

    prepared, manifest = load_prepared_splits(
        first["normalized_csv"],
        first["split_manifest"],
        subgroup_fields=["capture_mode", "pen_type"],
        require_training_ready=True,
    )
    for field in GROUP_FIELDS:
        values_by_split = [
            {record[field] for record in prepared[split_name]}
            for split_name in ("train", "calibration", "test")
        ]
        assert values_by_split[0].isdisjoint(values_by_split[1])
        assert values_by_split[0].isdisjoint(values_by_split[2])
        assert values_by_split[1].isdisjoint(values_by_split[2])

    model = train_serialized_calibrator(
        prepared["train"],
        prepared["calibration"],
        manifest["feature_names"],
        model_version="test-physical-v1",
        backend="numpy",
        calibration="sigmoid",
    )
    assert model.payload["model_type"] == "logistic_risk"
    assert model.payload["post_calibration"]["method"] == "sigmoid"


def test_capture_contract_rejects_duplicate_bytes_and_bad_mode(tmp_path):
    duplicate_source = _write_capture_csv(
        tmp_path / "duplicate", duplicate_content=True
    )
    with pytest.raises(
        DatasetContractError, match="Duplicate capture content"
    ):
        load_capture_rows(duplicate_source)

    invalid_root = tmp_path / "invalid"
    source = _write_capture_csv(invalid_root)
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    rows[0]["capture_mode"] = "camera"
    with source.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetContractError, match="capture_mode"):
        load_capture_rows(source)


def test_prepared_data_refuses_provenance_tampering(tmp_path):
    source = _write_capture_csv(tmp_path)
    prepared = prepare_capture_dataset(source, tmp_path / "prepared")
    with prepared["normalized_csv"].open("a", encoding="utf-8") as destination:
        destination.write("\n")

    with pytest.raises(DatasetContractError, match="SHA-256"):
        load_prepared_splits(
            prepared["normalized_csv"], prepared["split_manifest"]
        )


def test_classifier_gate_has_contract_but_no_fabricated_metrics():
    report = classifier_readiness_report()

    assert report["status"] == "blocked"
    assert report["benchmark_ready"] is False
    assert report["measured_metrics"] is None
    assert report["contract"]["export"]["format_version"] == (
        CLASSIFIER_EXPORT_SCHEMA_VERSION
    )
    assert "no verified bubble-crop manifest" in report["blocking_reasons"][0]


def test_classifier_crop_duplicates_fail_against_parent_splits(tmp_path):
    source = _write_capture_csv(tmp_path)
    prepared = prepare_capture_dataset(source, tmp_path / "prepared")
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    (crops_dir / "one.png").write_bytes(b"same-crop")
    (crops_dir / "two.png").write_bytes(b"same-crop")
    bubble_csv = tmp_path / "bubbles.csv"
    fields = list(BUBBLE_EXPORT_FIELDS)
    with bubble_csv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for index, capture_id in enumerate(
            ["capture-00-0", "capture-01-0"], start=1
        ):
            writer.writerow(
                {
                    "schema_version": BUBBLE_DATASET_SCHEMA_VERSION,
                    "crop_id": "crop-{}".format(index),
                    "capture_id": capture_id,
                    "physical_sheet_id": "sheet-0{}".format(index - 1),
                    "capture_session_id": "session-0{}".format(index - 1),
                    "device_id": "device-0{}".format(index - 1),
                    "print_batch_id": "batch-0{}".format(index - 1),
                    "capture_mode": (
                        "scanner" if index == 1 else "mobile"
                    ),
                    "question_key": "q1",
                    "option_key": "A",
                    "crop_path": "crops/{}.png".format(
                        "one" if index == 1 else "two"
                    ),
                    "crop_sha256": sha256_file(
                        crops_dir
                        / "{}.png".format(
                            "one" if index == 1 else "two"
                        )
                    ),
                    "label": "filled" if index == 1 else "empty",
                    "label_status": "verified",
                    "bbox_x": 1,
                    "bbox_y": 1,
                    "bbox_width": 8,
                    "bbox_height": 8,
                    "rectified_image_sha256": "0" * 64,
                    "processor_result_sha256": "1" * 64,
                    "human_annotations_sha256": "2" * 64,
                    "capture_metadata_sha256": "3" * 64,
                    "machine_label": (
                        "filled" if index == 1 else "empty"
                    ),
                    "machine_confidence": "0.9",
                }
            )

    with pytest.raises(DatasetContractError, match="Duplicate bubble content"):
        classifier_readiness_report(
            prepared["normalized_csv"],
            prepared["split_manifest"],
            bubble_csv,
        )
