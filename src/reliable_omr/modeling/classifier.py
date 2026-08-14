"""Gated OpenCV HOG + scikit-learn logistic bubble benchmark."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.reliable_omr.modeling.contracts import (
    CLASSIFIER_BENCHMARK_SCHEMA_VERSION,
    CLASSIFIER_EXPORT_REQUIRED_FIELDS,
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
)
from src.reliable_omr.modeling.dataset import (
    DatasetContractError,
    load_capture_rows,
    load_prepared_splits,
    sha256_file,
    sha256_lines,
)
from src.reliable_omr.modeling.readiness import (
    classifier_readiness_report,
    load_verified_bubble_rows,
)
from src.reliable_omr.opencv import require_cv2


IMAGE_SIZE = (32, 32)
HOG_PARAMETERS = {
    "name": "opencv_hog",
    "win_size": [32, 32],
    "block_size": [16, 16],
    "block_stride": [8, 8],
    "cell_size": [8, 8],
    "bins": 9,
}
DEFAULT_C_VALUES = (0.1, 1.0, 10.0)
_IDENTIFIER_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ClassifierContractError(DatasetContractError):
    """Raised when classifier data or numeric JSON is invalid."""


def _hog_descriptor():
    cv2 = require_cv2()
    return cv2.HOGDescriptor(
        tuple(HOG_PARAMETERS["win_size"]),
        tuple(HOG_PARAMETERS["block_size"]),
        tuple(HOG_PARAMETERS["block_stride"]),
        tuple(HOG_PARAMETERS["cell_size"]),
        int(HOG_PARAMETERS["bins"]),
    )


def normalize_grayscale_image(image: np.ndarray) -> np.ndarray:
    """Return deterministic 32x32 uint8 grayscale pixels."""

    cv2 = require_cv2()
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ClassifierContractError("Bubble image must be a non-empty array")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ClassifierContractError(
                "Bubble image has unsupported channel count"
            )
    elif image.ndim != 2:
        raise ClassifierContractError(
            "Bubble image must be grayscale, BGR, or BGRA"
        )
    if image.dtype != np.uint8:
        if not np.all(np.isfinite(image)):
            raise ClassifierContractError(
                "Bubble image contains non-finite pixels"
            )
        image = np.clip(image, 0, 255).astype(np.uint8)
    interpolation = (
        cv2.INTER_AREA
        if image.shape[0] > IMAGE_SIZE[1]
        or image.shape[1] > IMAGE_SIZE[0]
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, IMAGE_SIZE, interpolation=interpolation)


def extract_hog(image: np.ndarray) -> np.ndarray:
    normalized = normalize_grayscale_image(image)
    features = _hog_descriptor().compute(normalized)
    if features is None:
        raise ClassifierContractError("OpenCV HOG extraction failed")
    return features.reshape(-1).astype(np.float64)


def extract_hog_path(path: Path) -> np.ndarray:
    cv2 = require_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ClassifierContractError(
            "Bubble crop cannot be decoded: {}".format(path)
        )
    return extract_hog(image)


def _finite_vector(value: Any, field: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ClassifierContractError(
            "{} must be numeric".format(field)
        ) from exc
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ClassifierContractError(
            "{} must be a finite numeric vector".format(field)
        )
    return vector


def _finite_matrix(value: Any, field: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ClassifierContractError(
            "{} must be numeric with consistent dimensions".format(field)
        ) from exc
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ClassifierContractError(
            "{} must be a finite numeric matrix".format(field)
        )
    return matrix


class LightweightBubbleClassifier:
    """Portable numeric HOG/logistic model for offline shadow prediction."""

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        self._validate()

    def _validate(self) -> None:
        missing = sorted(
            set(CLASSIFIER_EXPORT_REQUIRED_FIELDS) - set(self.payload)
        )
        if missing:
            raise ClassifierContractError(
                "Classifier export is missing fields: {}".format(missing)
            )
        if (
            self.payload.get("format_version")
            != CLASSIFIER_EXPORT_SCHEMA_VERSION
        ):
            raise ClassifierContractError(
                "Unsupported classifier export format_version"
            )
        if self.payload.get("model_type") != "hog_multinomial_logistic":
            raise ClassifierContractError("Unsupported classifier model_type")
        model_version = self.payload.get("model_version")
        valid_version = (
            isinstance(model_version, str)
            and _IDENTIFIER_VERSION.fullmatch(model_version) is not None
        )
        if not valid_version:
            raise ClassifierContractError(
                "Classifier model_version is invalid"
            )
        if self.payload.get("image_size") != list(IMAGE_SIZE):
            raise ClassifierContractError(
                "Classifier image_size must be [32, 32]"
            )
        if self.payload.get("feature_extractor") != HOG_PARAMETERS:
            raise ClassifierContractError(
                "Classifier HOG parameters do not match the supported contract"
            )
        classes = self.payload.get("class_names")
        if (
            not isinstance(classes, list)
            or len(classes) < 2
            or len(classes) != len(set(classes))
            or any(not isinstance(item, str) or not item for item in classes)
        ):
            raise ClassifierContractError(
                "Classifier class_names must contain unique strings"
            )
        expected_features = int(_hog_descriptor().getDescriptorSize())
        means = _finite_vector(
            self.payload.get("feature_means"), "feature_means"
        )
        scales = _finite_vector(
            self.payload.get("feature_scales"), "feature_scales"
        )
        coefficients = _finite_matrix(
            self.payload.get("coefficients"), "coefficients"
        )
        intercepts = _finite_vector(
            self.payload.get("intercepts"), "intercepts"
        )
        if means.size != expected_features or scales.size != expected_features:
            raise ClassifierContractError(
                "Classifier scaling dimension does not match OpenCV HOG"
            )
        if np.any(scales <= 0):
            raise ClassifierContractError(
                "Classifier feature scales must be positive"
            )
        if coefficients.shape != (len(classes), expected_features):
            raise ClassifierContractError(
                "Classifier coefficient dimensions do not match classes/HOG"
            )
        if intercepts.size != len(classes):
            raise ClassifierContractError(
                "Classifier intercept dimensions do not match classes"
            )
        if not isinstance(self.payload.get("seed"), int):
            raise ClassifierContractError("Classifier seed must be an integer")
        if not isinstance(self.payload.get("selection"), dict):
            raise ClassifierContractError(
                "Classifier selection must be an object"
            )
        if not isinstance(self.payload.get("metrics"), dict):
            raise ClassifierContractError(
                "Classifier metrics must be an object"
            )
        provenance = self.payload.get("training_provenance")
        if not isinstance(provenance, dict):
            raise ClassifierContractError(
                "Classifier training_provenance must be an object"
            )
        required_hashes = {
            "normalized_capture_csv_sha256",
            "split_manifest_sha256",
            "bubble_manifest_sha256",
            "train_crop_set_sha256",
            "calibration_crop_set_sha256",
            "test_crop_set_sha256",
        }
        for field in required_hashes:
            value = provenance.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ClassifierContractError(
                    "Classifier provenance {} must be SHA-256".format(field)
                )
        if self.payload.get("deployment_status") != "shadow_only":
            raise ClassifierContractError(
                "Classifier export must remain shadow_only"
            )
        self.class_names = list(classes)
        self.feature_means = means
        self.feature_scales = scales
        self.coefficients = coefficients
        self.intercepts = intercepts

    @classmethod
    def load(cls, path: Path) -> "LightweightBubbleClassifier":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ClassifierContractError(
                "Unable to read classifier JSON: {}".format(exc)
            ) from exc
        if not isinstance(payload, dict):
            raise ClassifierContractError(
                "Classifier JSON must contain an object"
            )
        return cls(payload)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def predict_features(self, features: np.ndarray) -> List[Dict[str, Any]]:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.feature_means.size
            or not np.all(np.isfinite(matrix))
        ):
            raise ClassifierContractError(
                "Prediction features do not match classifier HOG dimensions"
            )
        scaled = (matrix - self.feature_means) / self.feature_scales
        logits = scaled.dot(self.coefficients.T) + self.intercepts
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        output = []
        for row in probabilities:
            index = int(np.argmax(row))
            output.append(
                {
                    "label": self.class_names[index],
                    "confidence": float(row[index]),
                    "probabilities": {
                        name: float(row[class_index])
                        for class_index, name in enumerate(self.class_names)
                    },
                }
            )
        return output

    def predict_images(
        self, paths: Sequence[Path]
    ) -> List[Dict[str, Any]]:
        if not paths:
            return []
        features = np.vstack([extract_hog_path(Path(path)) for path in paths])
        return self.predict_features(features)

    def shadow_predict(
        self,
        paths: Sequence[Path],
        machine_labels: Optional[Sequence[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        predictions = self.predict_images(paths)
        if machine_labels is not None and len(machine_labels) != len(paths):
            raise ClassifierContractError(
                "machine_labels length must match image paths"
            )
        output = []
        for index, prediction in enumerate(predictions):
            output.append(
                {
                    "deployment_status": "shadow_only",
                    "classifier": prediction,
                    "machine_label": (
                        machine_labels[index]
                        if machine_labels is not None
                        else None
                    ),
                }
            )
        return output


def _classification_metrics(
    truth: Sequence[str],
    predicted: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ClassifierContractError(
            "Metrics require equal non-empty truth/prediction values"
        )
    index = {name: position for position, name in enumerate(class_names)}
    confusion = [
        [0 for _ in class_names]
        for _ in class_names
    ]
    for actual, guess in zip(truth, predicted):
        if actual not in index or guess not in index:
            raise ClassifierContractError(
                "Metric label is absent from classifier class_names"
            )
        confusion[index[actual]][index[guess]] += 1
    per_class = {}
    recalls = []
    for class_index, name in enumerate(class_names):
        true_positive = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted_count = sum(row[class_index] for row in confusion)
        precision = (
            true_positive / float(predicted_count)
            if predicted_count
            else 0.0
        )
        recall = true_positive / float(support) if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        if support:
            recalls.append(recall)
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    correct = sum(
        confusion[class_index][class_index]
        for class_index in range(len(class_names))
    )
    return {
        "sample_count": len(truth),
        "accuracy": correct / float(len(truth)),
        "balanced_accuracy": (
            float(sum(recalls) / len(recalls)) if recalls else 0.0
        ),
        "class_names": list(class_names),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _crop_set_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_lines(
        "{}:{}:{}".format(
            row["crop_id"], row["capture_id"], row["crop_sha256"]
        )
        for row in rows
    )


def _load_classifier_rows(
    normalized_csv: Path,
    split_manifest: Path,
    bubble_csv: Path,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    prepared, manifest = load_prepared_splits(
        normalized_csv, split_manifest, require_training_ready=False
    )
    captures_list, _ = load_capture_rows(normalized_csv)
    captures = {
        str(row["capture_id"]): row for row in captures_list
    }
    assignment = {
        str(record["capture_id"]): split_name
        for split_name, records in prepared.items()
        for record in records
    }
    rows = load_verified_bubble_rows(bubble_csv, captures, assignment)
    by_split = {
        split_name: [
            row for row in rows if row["split"] == split_name
        ]
        for split_name in ("train", "calibration", "test")
    }
    if any(not by_split[name] for name in by_split):
        raise ClassifierContractError(
            "Every capture-level split must contain bubble crops"
        )
    train_classes = sorted({str(row["label"]) for row in by_split["train"]})
    if len(train_classes) < 2:
        raise ClassifierContractError(
            "Classifier training requires at least two human-labelled classes"
        )
    for split_name, rows_for_split in by_split.items():
        split_classes = {str(row["label"]) for row in rows_for_split}
        if split_classes != set(train_classes):
            raise ClassifierContractError(
                "{} bubble split must contain exactly the train classes; "
                "expected {} found {}".format(
                    split_name, train_classes, sorted(split_classes)
                )
            )
    return by_split, manifest


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.vstack(
        [
            extract_hog_path(Path(row["resolved_crop_path"]))
            for row in rows
        ]
    )
    labels = np.asarray([str(row["label"]) for row in rows])
    return features, labels


def _export_coefficients(
    estimator: Any,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    classes = [str(value) for value in estimator.classes_.tolist()]
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    intercepts = np.asarray(estimator.intercept_, dtype=np.float64)
    if len(classes) == 2 and coefficients.shape[0] == 1:
        coefficients = np.vstack(
            [np.zeros(coefficients.shape[1]), coefficients[0]]
        )
        intercepts = np.asarray([0.0, intercepts[0]], dtype=np.float64)
    return classes, coefficients, intercepts


def train_bubble_classifier(
    normalized_csv: Path,
    split_manifest: Path,
    bubble_csv: Path,
    output_model: Path,
    model_version: str,
    seed: int = 2026,
    c_values: Sequence[float] = DEFAULT_C_VALUES,
) -> Dict[str, Any]:
    """Train only after the existing readiness gate passes."""

    readiness = classifier_readiness_report(
        normalized_csv, split_manifest, bubble_csv
    )
    if not readiness["benchmark_ready"]:
        raise ClassifierContractError(
            "Classifier readiness gate is blocked: {}".format(
                "; ".join(readiness["blocking_reasons"])
            )
        )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ClassifierContractError(
            "scikit-learn is required for classifier training"
        ) from exc

    if not _IDENTIFIER_VERSION.fullmatch(str(model_version)):
        raise ClassifierContractError(
            "model_version must use letters, numbers, '.', '_', or '-'"
        )
    candidates = sorted({float(value) for value in c_values})
    if (
        not candidates
        or any(not math.isfinite(value) or value <= 0 for value in candidates)
    ):
        raise ClassifierContractError(
            "All logistic C candidates must be finite and positive"
        )

    by_split, manifest = _load_classifier_rows(
        normalized_csv, split_manifest, bubble_csv
    )
    train_features, train_labels = _feature_matrix(by_split["train"])
    calibration_features, calibration_labels = _feature_matrix(
        by_split["calibration"]
    )
    test_features, test_labels = _feature_matrix(by_split["test"])
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_calibration = scaler.transform(calibration_features)
    scaled_test = scaler.transform(test_features)

    selected = None
    selection_scores = []
    for c_value in candidates:
        estimator = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            random_state=int(seed),
        )
        estimator.fit(scaled_train, train_labels)
        calibration_predicted = estimator.predict(
            scaled_calibration
        ).tolist()
        classes = [str(value) for value in estimator.classes_.tolist()]
        metrics = _classification_metrics(
            calibration_labels.tolist(), calibration_predicted, classes
        )
        score = float(metrics["balanced_accuracy"])
        selection_scores.append(
            {"c": float(c_value), "balanced_accuracy": score}
        )
        if selected is None or score > selected[0]:
            selected = (score, c_value, estimator)
    assert selected is not None
    estimator = selected[2]
    class_names, coefficients, intercepts = _export_coefficients(estimator)
    test_predicted = estimator.predict(scaled_test).tolist()
    test_metrics = _classification_metrics(
        test_labels.tolist(), test_predicted, class_names
    )
    provenance = {
        "normalized_capture_csv_sha256": sha256_file(normalized_csv),
        "split_manifest_sha256": sha256_file(split_manifest),
        "bubble_manifest_sha256": sha256_file(bubble_csv),
        "train_crop_set_sha256": _crop_set_hash(by_split["train"]),
        "calibration_crop_set_sha256": _crop_set_hash(
            by_split["calibration"]
        ),
        "test_crop_set_sha256": _crop_set_hash(by_split["test"]),
    }
    payload = {
        "format_version": CLASSIFIER_EXPORT_SCHEMA_VERSION,
        "model_type": "hog_multinomial_logistic",
        "model_version": str(model_version),
        "class_names": class_names,
        "image_size": list(IMAGE_SIZE),
        "feature_extractor": dict(HOG_PARAMETERS),
        "feature_means": np.asarray(
            scaler.mean_, dtype=np.float64
        ).tolist(),
        "feature_scales": np.asarray(
            scaler.scale_, dtype=np.float64
        ).tolist(),
        "coefficients": coefficients.tolist(),
        "intercepts": intercepts.tolist(),
        "seed": int(seed),
        "training_provenance": provenance,
        "selection": {
            "split": "calibration",
            "metric": "balanced_accuracy",
            "selected_c": float(selected[1]),
            "candidate_scores": selection_scores,
            "probability_calibration": "not_applied",
        },
        "metrics": {
            "split": "test",
            "multiclass": test_metrics,
            "test_was_used_for_fitting_or_selection": False,
        },
        "deployment_status": "shadow_only",
    }
    model = LightweightBubbleClassifier(payload)
    model.save(output_model)
    return {
        "model": model,
        "model_path": Path(output_model),
        "readiness": readiness,
        "metrics": payload["metrics"],
    }


def _risk_coverage(
    truth_filled: Sequence[bool],
    predicted_filled: Sequence[bool],
    confidence: Sequence[float],
    thresholds: Sequence[float] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95),
    total_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not (
        len(truth_filled) == len(predicted_filled) == len(confidence)
    ):
        raise ClassifierContractError(
            "Risk/coverage inputs must have equal lengths"
        )
    total = len(truth_filled) if total_count is None else int(total_count)
    if total < len(truth_filled):
        raise ClassifierContractError(
            "Risk/coverage total_count cannot be below available predictions"
        )
    output = []
    for threshold in thresholds:
        selected = [
            index
            for index, value in enumerate(confidence)
            if float(value) >= threshold
        ]
        errors = sum(
            predicted_filled[index] != truth_filled[index]
            for index in selected
        )
        output.append(
            {
                "confidence_threshold": float(threshold),
                "covered_count": len(selected),
                "coverage": len(selected) / float(total) if total else 0.0,
                "error_count": int(errors),
                "risk": (
                    errors / float(len(selected)) if selected else None
                ),
            }
        )
    return output


def evaluate_bubble_classifier(
    model: Union[Path, LightweightBubbleClassifier],
    normalized_csv: Path,
    split_manifest: Path,
    bubble_csv: Path,
) -> Dict[str, Any]:
    """Evaluate a frozen model on the untouched capture-level test split."""

    loaded = (
        model
        if isinstance(model, LightweightBubbleClassifier)
        else LightweightBubbleClassifier.load(Path(model))
    )
    by_split, _ = _load_classifier_rows(
        normalized_csv, split_manifest, bubble_csv
    )
    expected_provenance = {
        "normalized_capture_csv_sha256": sha256_file(normalized_csv),
        "split_manifest_sha256": sha256_file(split_manifest),
        "bubble_manifest_sha256": sha256_file(bubble_csv),
        "train_crop_set_sha256": _crop_set_hash(by_split["train"]),
        "calibration_crop_set_sha256": _crop_set_hash(
            by_split["calibration"]
        ),
        "test_crop_set_sha256": _crop_set_hash(by_split["test"]),
    }
    if loaded.payload["training_provenance"] != expected_provenance:
        raise ClassifierContractError(
            "Classifier provenance does not match dataset/split/crops"
        )
    test_rows = by_split["test"]
    test_features, test_labels = _feature_matrix(test_rows)
    predictions = loaded.predict_features(test_features)
    predicted_labels = [str(item["label"]) for item in predictions]
    multiclass = _classification_metrics(
        test_labels.tolist(), predicted_labels, loaded.class_names
    )
    by_mode = {}
    for mode in ("scanner", "mobile"):
        indexes = [
            index
            for index, row in enumerate(test_rows)
            if row["capture_mode"] == mode
        ]
        if indexes:
            by_mode[mode] = _classification_metrics(
                [str(test_labels[index]) for index in indexes],
                [predicted_labels[index] for index in indexes],
                loaded.class_names,
            )
        else:
            by_mode[mode] = {"sample_count": 0, "status": "insufficient_data"}

    truth_filled = [label == "filled" for label in test_labels.tolist()]
    classifier_filled = [
        prediction["label"] == "filled" for prediction in predictions
    ]
    classifier_confidence = []
    for prediction in predictions:
        filled_probability = float(
            prediction["probabilities"].get("filled", 0.0)
        )
        classifier_confidence.append(
            max(filled_probability, 1.0 - filled_probability)
        )
    baseline_indexes = [
        index
        for index, row in enumerate(test_rows)
        if row.get("machine_label") in {"empty", "filled"}
    ]
    baseline_errors = sum(
        (test_rows[index]["machine_label"] == "filled")
        != truth_filled[index]
        for index in baseline_indexes
    )
    baseline = {
        "available_count": len(baseline_indexes),
        "coverage": len(baseline_indexes) / float(len(test_rows)),
        "error_count": int(baseline_errors),
        "risk": (
            baseline_errors / float(len(baseline_indexes))
            if baseline_indexes
            else None
        ),
        "risk_coverage": _risk_coverage(
            [truth_filled[index] for index in baseline_indexes],
            [
                test_rows[index]["machine_label"] == "filled"
                for index in baseline_indexes
            ],
            [
                float(test_rows[index]["machine_confidence"])
                for index in baseline_indexes
            ],
            total_count=len(test_rows),
        )
        if baseline_indexes
        else [],
    }
    return {
        "schema_version": CLASSIFIER_BENCHMARK_SCHEMA_VERSION,
        "model_version": loaded.payload["model_version"],
        "deployment_status": "shadow_only",
        "evaluation_split": "test",
        "test_was_used_for_fitting_or_selection": False,
        "provenance": expected_provenance,
        "multiclass": multiclass,
        "by_capture_mode": by_mode,
        "filled_vs_not": {
            "classifier": {
                "risk_coverage": _risk_coverage(
                    truth_filled,
                    classifier_filled,
                    classifier_confidence,
                )
            },
            "deterministic_machine_baseline": baseline,
        },
    }
