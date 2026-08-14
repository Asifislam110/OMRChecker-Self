"""Transparent heuristic risk routing and portable logistic calibration."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.reliable_omr.types import (
    ConfidenceDiagnostics,
    MarkStatus,
    QualityReport,
    QuestionResult,
    RectificationDiagnostics,
)


RISK_FEATURES = [
    "quality_error_count",
    "quality_review_count",
    "rectification_fallback",
    "reprojection_error_norm",
    "ambiguous_fraction",
    "multiple_fraction",
    "invalid_fraction",
    "low_margin_fraction",
    "mean_question_confidence",
    "min_question_confidence",
    "roll_uncertain_fraction",
    "qr_missing",
]


class CalibratorError(ValueError):
    pass


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(max(value, -700.0))
    return exp_value / (1.0 + exp_value)


def build_risk_features(
    questions: Sequence[QuestionResult],
    roll_columns: Sequence[QuestionResult],
    quality: QualityReport,
    rectification: RectificationDiagnostics,
    qr_payload: Optional[Mapping[str, Any]],
    definition: Mapping[str, Any],
) -> Dict[str, float]:
    count = max(len(questions), 1)
    margins = [question.margin for question in questions]
    confidences = [question.confidence for question in questions] or [0.0]
    min_margin = float(definition["classification"]["min_top_margin"])
    roll_count = max(len(roll_columns), 1)
    return {
        "quality_error_count": float(
            sum(issue.severity == "error" for issue in quality.issues)
        ),
        "quality_review_count": float(
            sum(issue.severity == "review" for issue in quality.issues)
        ),
        "rectification_fallback": float(
            rectification.method != "aruco_homography"
        ),
        "reprojection_error_norm": float(
            (rectification.reprojection_error_px or 0.0)
            / max(
                float(definition["quality"]["max_reprojection_error_px"]),
                1e-6,
            )
        ),
        "ambiguous_fraction": sum(
            question.status == MarkStatus.AMBIGUOUS for question in questions
        )
        / float(count),
        "multiple_fraction": sum(
            question.status == MarkStatus.MULTIPLE for question in questions
        )
        / float(count),
        "invalid_fraction": sum(
            question.status == MarkStatus.INVALID for question in questions
        )
        / float(count),
        "low_margin_fraction": sum(margin < min_margin for margin in margins)
        / float(count),
        "mean_question_confidence": float(np.mean(confidences)),
        "min_question_confidence": float(np.min(confidences)),
        "roll_uncertain_fraction": sum(
            column.status != MarkStatus.FILLED for column in roll_columns
        )
        / float(roll_count),
        "qr_missing": float(qr_payload is None),
    }


def heuristic_risk(
    features: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Return a documented additive risk score and per-feature contributions."""

    contributions = {
        "quality_errors": min(0.55, 0.35 * features["quality_error_count"]),
        "quality_reviews": min(0.18, 0.09 * features["quality_review_count"]),
        "rectification_fallback": 0.25 * features["rectification_fallback"],
        "reprojection_error": min(
            0.18, 0.12 * features["reprojection_error_norm"]
        ),
        "ambiguous_questions": 0.5 * features["ambiguous_fraction"],
        "multiple_questions": 0.65 * features["multiple_fraction"],
        "invalid_questions": 0.8 * features["invalid_fraction"],
        "low_margins": 0.22 * features["low_margin_fraction"],
        "low_mean_confidence": 0.2
        * max(0.0, 0.8 - features["mean_question_confidence"]),
        "low_min_confidence": 0.1
        * max(0.0, 0.6 - features["min_question_confidence"]),
        "roll_number": 0.12 * features["roll_uncertain_fraction"],
        "qr_missing": 0.04 * features["qr_missing"],
    }
    return min(1.0, float(sum(contributions.values()))), contributions


class SerializedRiskCalibrator:
    """A JSON-only logistic model with optional sigmoid/isotonic calibration."""

    FORMAT_VERSION = 1

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = dict(payload)
        self._validate()

    def _validate(self) -> None:
        required = {
            "format_version",
            "model_type",
            "model_version",
            "feature_names",
            "means",
            "scales",
            "coefficients",
            "intercept",
        }
        missing = sorted(required - set(self.payload))
        if missing:
            raise CalibratorError(
                "Calibrator is missing fields: {}".format(missing)
            )
        if self.payload["format_version"] != self.FORMAT_VERSION:
            raise CalibratorError("Unsupported calibrator format version")
        if self.payload["model_type"] != "logistic_risk":
            raise CalibratorError("Only logistic_risk calibrators are supported")
        lengths = [
            len(self.payload[name])
            for name in ("feature_names", "means", "scales", "coefficients")
        ]
        if len(set(lengths)) != 1 or not lengths[0]:
            raise CalibratorError(
                "Feature names and logistic parameter lengths must match"
            )
        feature_names = self.payload["feature_names"]
        if len(set(feature_names)) != len(feature_names):
            raise CalibratorError("Calibrator feature names must be unique")
        unsupported = sorted(set(feature_names) - set(RISK_FEATURES))
        if unsupported:
            raise CalibratorError(
                "Calibrator has unsupported risk features: {}".format(
                    unsupported
                )
            )
        if any(float(scale) <= 0 for scale in self.payload["scales"]):
            raise CalibratorError("Calibrator scales must be positive")
        numeric_parameters = [
            *self.payload["means"],
            *self.payload["scales"],
            *self.payload["coefficients"],
            self.payload["intercept"],
        ]
        if not all(math.isfinite(float(value)) for value in numeric_parameters):
            raise CalibratorError("Calibrator parameters must be finite")
        post = self.payload.get("post_calibration", {"method": "none"})
        if post.get("method") not in {"none", "sigmoid", "isotonic"}:
            raise CalibratorError("Unknown post-calibration method")
        if post.get("method") == "isotonic":
            x_values = post.get("x", [])
            y_values = post.get("y", [])
            if len(x_values) != len(y_values) or len(x_values) < 2:
                raise CalibratorError(
                    "Isotonic x/y lengths must match and contain two points"
                )
            if list(x_values) != sorted(x_values):
                raise CalibratorError("Isotonic x values must be sorted")

    @classmethod
    def load(
        cls, source: Union[str, Path, Mapping[str, Any]]
    ) -> "SerializedRiskCalibrator":
        if isinstance(source, Mapping):
            return cls(source)
        if isinstance(source, str) and source.lstrip().startswith("{"):
            return cls(json.loads(source))
        with Path(source).open("r", encoding="utf-8") as model_file:
            return cls(json.load(model_file))

    def save(self, path: Union[str, Path]) -> None:
        with Path(path).open("w", encoding="utf-8") as model_file:
            json.dump(self.payload, model_file, indent=2, sort_keys=True)
            model_file.write("\n")

    def predict(self, features: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
        names = self.payload["feature_names"]
        values = np.asarray(
            [float(features.get(name, 0.0)) for name in names],
            dtype=np.float64,
        )
        means = np.asarray(self.payload["means"], dtype=np.float64)
        scales = np.asarray(self.payload["scales"], dtype=np.float64)
        coefficients = np.asarray(
            self.payload["coefficients"], dtype=np.float64
        )
        standardized = (values - means) / scales
        terms = standardized * coefficients
        logit = float(self.payload["intercept"] + np.sum(terms))
        probability = _sigmoid(logit)
        post = self.payload.get("post_calibration", {"method": "none"})
        if post.get("method") == "sigmoid":
            probability = _sigmoid(
                float(post["a"]) * logit + float(post["b"])
            )
        elif post.get("method") == "isotonic":
            probability = float(
                np.interp(
                    probability,
                    np.asarray(post["x"], dtype=np.float64),
                    np.asarray(post["y"], dtype=np.float64),
                )
            )
        contributions = {
            name: round(float(term), 6)
            for name, term in zip(names, terms)
        }
        contributions["intercept"] = round(
            float(self.payload["intercept"]), 6
        )
        return float(np.clip(probability, 0.0, 1.0)), contributions


def route_risk(
    features: Mapping[str, float],
    calibrator: Optional[SerializedRiskCalibrator],
) -> ConfidenceDiagnostics:
    baseline, heuristic_contributions = heuristic_risk(features)
    if calibrator is None:
        risk = baseline
        source = "heuristic"
        contributions = heuristic_contributions
        version = None
    else:
        risk, contributions = calibrator.predict(features)
        source = "serialized_calibrator"
        version = str(calibrator.payload["model_version"])
    return ConfidenceDiagnostics(
        source=source,
        risk=round(risk, 6),
        heuristic_risk=round(baseline, 6),
        features={key: round(float(value), 6) for key, value in features.items()},
        contributions={
            key: round(float(value), 6)
            for key, value in contributions.items()
        },
        calibrator_version=version,
    )


def train_logistic_model(
    rows: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    feature_names: Sequence[str] = RISK_FEATURES,
    model_version: str = "1",
    l2: float = 0.01,
    iterations: int = 2500,
    learning_rate: float = 0.05,
) -> Dict[str, Any]:
    """Train a small NumPy logistic model without adding a core ML dependency."""

    if len(rows) != len(labels) or len(rows) < 4:
        raise CalibratorError("Training requires at least four labelled rows")
    y = np.asarray(labels, dtype=np.float64)
    if set(np.unique(y)) != {0.0, 1.0}:
        raise CalibratorError("Training labels must include both 0 and 1")
    x = np.asarray(
        [
            [float(row.get(name, 0.0)) for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    normalized = (x - means) / scales
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    positive_rate = float(np.clip(y.mean(), 1e-5, 1 - 1e-5))
    intercept = math.log(positive_rate / (1.0 - positive_rate))

    for _ in range(iterations):
        logits = np.clip(normalized.dot(weights) + intercept, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y
        weight_gradient = normalized.T.dot(error) / len(y) + l2 * weights
        intercept_gradient = float(error.mean())
        weights -= learning_rate * weight_gradient
        intercept -= learning_rate * intercept_gradient

    return {
        "format_version": SerializedRiskCalibrator.FORMAT_VERSION,
        "model_type": "logistic_risk",
        "model_version": str(model_version),
        "feature_names": list(feature_names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": weights.tolist(),
        "intercept": float(intercept),
        "post_calibration": {"method": "none"},
    }


def fit_sigmoid_calibration(
    probabilities: Sequence[float],
    labels: Sequence[int],
    iterations: int = 1500,
    learning_rate: float = 0.03,
) -> Dict[str, Any]:
    """Fit Platt-style sigmoid parameters on a disjoint calibration set."""

    if len(probabilities) != len(labels) or len(probabilities) < 4:
        raise CalibratorError("Calibration requires at least four labelled rows")
    y = np.asarray(labels, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise CalibratorError("Calibration labels must include both classes")
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    a, b = 1.0, 0.0
    for _ in range(iterations):
        predicted = 1.0 / (1.0 + np.exp(-np.clip(a * logits + b, -30, 30)))
        error = predicted - y
        a -= learning_rate * float(np.mean(error * logits))
        b -= learning_rate * float(np.mean(error))
    return {"method": "sigmoid", "a": float(a), "b": float(b)}
