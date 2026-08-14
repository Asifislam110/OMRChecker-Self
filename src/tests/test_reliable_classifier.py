import csv
import importlib.util
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.reliable_omr.calibration import RISK_FEATURES  # noqa: E402
from src.reliable_omr.modeling.classifier import (  # noqa: E402
    ClassifierContractError,
    HOG_PARAMETERS,
    IMAGE_SIZE,
    LightweightBubbleClassifier,
    evaluate_bubble_classifier,
    extract_hog,
    train_bubble_classifier,
)
from src.reliable_omr.modeling.contracts import (  # noqa: E402
    BUBBLE_DATASET_SCHEMA_VERSION,
    BUBBLE_EXPORT_FIELDS,
    CAPTURE_REQUIRED_FIELDS,
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
)
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
    prepare_capture_dataset,
    sha256_file,
)
from src.reliable_omr.modeling import readiness  # noqa: E402


SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def _portable_payload(feature_count):
    return {
        "format_version": CLASSIFIER_EXPORT_SCHEMA_VERSION,
        "model_type": "hog_multinomial_logistic",
        "model_version": "roundtrip-v1",
        "class_names": ["empty", "filled"],
        "image_size": list(IMAGE_SIZE),
        "feature_extractor": dict(HOG_PARAMETERS),
        "feature_means": [0.0] * feature_count,
        "feature_scales": [1.0] * feature_count,
        "coefficients": [
            [0.0] * feature_count,
            [0.001] * feature_count,
        ],
        "intercepts": [0.0, -0.1],
        "seed": 7,
        "training_provenance": {
            "normalized_capture_csv_sha256": "0" * 64,
            "split_manifest_sha256": "1" * 64,
            "bubble_manifest_sha256": "2" * 64,
            "train_crop_set_sha256": "3" * 64,
            "calibration_crop_set_sha256": "4" * 64,
            "test_crop_set_sha256": "5" * 64,
        },
        "selection": {
            "selected_c": 1.0,
            "candidate_scores": [{"c": 1.0, "balanced_accuracy": 0.5}],
        },
        "metrics": {"balanced_accuracy": 0.5},
        "deployment_status": "shadow_only",
    }


def test_classifier_json_round_trip_and_shadow_inference(tmp_path):
    sample = np.full((45, 45), 255, dtype=np.uint8)
    cv2.circle(sample, (22, 22), 10, 0, -1)
    features = extract_hog(sample)
    model = LightweightBubbleClassifier(_portable_payload(features.size))
    before = model.predict_features(features)[0]
    model_path = tmp_path / "model.json"
    model.save(model_path)

    loaded = LightweightBubbleClassifier.load(model_path)
    after = loaded.predict_features(features)[0]
    shadow = loaded.shadow_predict(
        [tmp_path / "crop.png"], ["empty"]
    ) if cv2.imwrite(str(tmp_path / "crop.png"), sample) else []

    assert after["label"] == before["label"]
    assert after["probabilities"] == pytest.approx(before["probabilities"])
    assert shadow[0]["deployment_status"] == "shadow_only"
    assert shadow[0]["machine_label"] == "empty"
    assert shadow[0]["classifier"]["label"] in {"empty", "filled"}


def test_classifier_loader_rejects_dimension_tampering():
    feature_count = extract_hog(np.zeros((32, 32), dtype=np.uint8)).size
    payload = _portable_payload(feature_count)
    payload["coefficients"][0].pop()

    with pytest.raises(ClassifierContractError, match="dimensions"):
        LightweightBubbleClassifier(payload)


def _capture_features(is_error):
    values = {name: 0.0 for name in RISK_FEATURES}
    values["mean_question_confidence"] = 0.4 if is_error else 0.95
    values["min_question_confidence"] = 0.2 if is_error else 0.85
    values["ambiguous_fraction"] = 0.2 if is_error else 0.0
    return values


def _write_classifier_dataset(tmp_path):
    image_dir = tmp_path / "captures"
    image_dir.mkdir(parents=True)
    capture_rows = []
    for index in range(30):
        mode = "scanner" if index % 2 == 0 else "mobile"
        capture_id = "capture-{:02d}".format(index)
        image_path = image_dir / "{}.bin".format(capture_id)
        image_path.write_bytes(
            "unique-capture-{}".format(index).encode("ascii")
        )
        row = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "capture_id": capture_id,
            "image_path": "captures/{}.bin".format(capture_id),
            "physical_sheet_id": "sheet-{:02d}".format(index),
            "capture_session_id": "session-{:02d}".format(index),
            "device_id": "device-{:02d}".format(index),
            "print_batch_id": "batch-{:02d}".format(index),
            "capture_mode": mode,
            "is_error": index % 2,
            "label_status": "verified",
        }
        row.update(_capture_features(index % 2))
        capture_rows.append(row)
    capture_csv = tmp_path / "captures.csv"
    with capture_csv.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=[*CAPTURE_REQUIRED_FIELDS, *RISK_FEATURES],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(capture_rows)
    prepared = prepare_capture_dataset(
        capture_csv, tmp_path / "prepared", seed=17
    )

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    bubble_csv = tmp_path / "bubble_crops.csv"
    with bubble_csv.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(BUBBLE_EXPORT_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, capture in enumerate(capture_rows):
            for label_index, label in enumerate(("empty", "filled")):
                crop_id = "{}.q1.{}".format(
                    capture["capture_id"], "A" if label_index == 0 else "B"
                )
                crop = np.full((48, 48), 245, dtype=np.uint8)
                cv2.circle(crop, (24, 24), 12, 30, 2)
                if label == "filled":
                    cv2.circle(crop, (24, 24), 9, 20, -1)
                crop[index % 48, (index * 7 + label_index) % 48] = index
                crop_path = crops_dir / "{}.png".format(crop_id)
                assert cv2.imwrite(str(crop_path), crop)
                writer.writerow(
                    {
                        "schema_version": BUBBLE_DATASET_SCHEMA_VERSION,
                        "crop_id": crop_id,
                        "capture_id": capture["capture_id"],
                        "physical_sheet_id": capture["physical_sheet_id"],
                        "capture_session_id": capture[
                            "capture_session_id"
                        ],
                        "device_id": capture["device_id"],
                        "print_batch_id": capture["print_batch_id"],
                        "capture_mode": capture["capture_mode"],
                        "question_key": "q1",
                        "option_key": "A" if label_index == 0 else "B",
                        "crop_path": "crops/{}.png".format(crop_id),
                        "crop_sha256": sha256_file(crop_path),
                        "label": label,
                        "label_status": "verified",
                        "bbox_x": 10,
                        "bbox_y": 10,
                        "bbox_width": 48,
                        "bbox_height": 48,
                        "rectified_image_sha256": "0" * 64,
                        "processor_result_sha256": "1" * 64,
                        "human_annotations_sha256": "2" * 64,
                        "capture_metadata_sha256": "3" * 64,
                        "machine_label": label,
                        "machine_confidence": "0.9",
                    }
                )
    return prepared, bubble_csv


def _lower_readiness_gates(monkeypatch):
    monkeypatch.setattr(readiness, "MIN_CLASSIFIER_CROPS_PER_LABEL_MODE", 1)
    monkeypatch.setattr(readiness, "MIN_CLASSIFIER_SHEETS_PER_MODE", 1)
    monkeypatch.setattr(
        readiness, "MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE", 1
    )


@pytest.mark.skipif(
    not SKLEARN_AVAILABLE, reason="optional scikit-learn is not installed"
)
def test_gated_train_evaluate_and_provenance_refusal(tmp_path, monkeypatch):
    prepared, bubble_csv = _write_classifier_dataset(tmp_path)
    _lower_readiness_gates(monkeypatch)
    model_path = tmp_path / "classifier.json"

    result = train_bubble_classifier(
        prepared["normalized_csv"],
        prepared["split_manifest"],
        bubble_csv,
        model_path,
        model_version="fixture-v1",
        seed=19,
        c_values=(0.1, 1.0),
    )
    report = evaluate_bubble_classifier(
        model_path,
        prepared["normalized_csv"],
        prepared["split_manifest"],
        bubble_csv,
    )

    assert result["model"].payload["deployment_status"] == "shadow_only"
    assert result["model"].payload["selection"]["split"] == "calibration"
    assert result["model"].payload["metrics"]["split"] == "test"
    assert report["multiclass"]["sample_count"] > 0
    assert set(report["multiclass"]["per_class"]) == {"empty", "filled"}
    assert set(report["by_capture_mode"]) == {"scanner", "mobile"}
    assert report["filled_vs_not"]["deterministic_machine_baseline"][
        "coverage"
    ] == pytest.approx(1.0)

    with bubble_csv.open("a", encoding="utf-8") as destination:
        destination.write("\n")
    with pytest.raises(ClassifierContractError, match="provenance"):
        evaluate_bubble_classifier(
            model_path,
            prepared["normalized_csv"],
            prepared["split_manifest"],
            bubble_csv,
        )


def test_training_refuses_blocked_readiness_and_split_tampering(
    tmp_path, monkeypatch
):
    prepared, bubble_csv = _write_classifier_dataset(tmp_path)
    with pytest.raises(ClassifierContractError, match="readiness gate"):
        train_bubble_classifier(
            prepared["normalized_csv"],
            prepared["split_manifest"],
            bubble_csv,
            tmp_path / "blocked.json",
            model_version="blocked-v1",
        )

    manifest = json.loads(
        prepared["split_manifest"].read_text(encoding="utf-8")
    )
    manifest["group_fields"] = ["physical_sheet_id"]
    prepared["split_manifest"].write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _lower_readiness_gates(monkeypatch)
    with pytest.raises(DatasetContractError, match="omits required leakage"):
        readiness.classifier_readiness_report(
            prepared["normalized_csv"],
            prepared["split_manifest"],
            bubble_csv,
        )
