"""Leakage-safe sheet-level data preparation for risk calibration scripts."""

import csv
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.reliable_omr.calibration import (
    CalibratorError,
    SerializedRiskCalibrator,
    fit_sigmoid_calibration,
    train_logistic_model,
)


def load_sheet_records(
    csv_path: Path,
    feature_names: Sequence[str],
    sheet_id_field: str = "sheet_id",
    label_field: str = "is_error",
    subgroup_fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Aggregate repeated rows before splitting, preventing sheet leakage."""

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        raw_rows = list(csv.DictReader(csv_file))
    if not raw_rows:
        raise CalibratorError("Training CSV contains no records")
    required = {sheet_id_field, label_field, *feature_names}
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise CalibratorError("Training CSV is missing columns: {}".format(missing))

    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in raw_rows:
        sheet_id = str(row[sheet_id_field]).strip()
        if not sheet_id:
            raise CalibratorError("Every row must have a non-empty sheet_id")
        grouped.setdefault(sheet_id, []).append(row)

    records = []
    for sheet_id, rows in sorted(grouped.items()):
        labels = [_parse_binary(row[label_field], label_field) for row in rows]
        record: Dict[str, Any] = {
            "sheet_id": sheet_id,
            "is_error": max(labels),
            "source_row_count": len(rows),
        }
        for feature in feature_names:
            try:
                record[feature] = float(
                    np.mean([float(row[feature]) for row in rows])
                )
            except (TypeError, ValueError) as exc:
                raise CalibratorError(
                    "Feature '{}' must be numeric for sheet '{}'".format(
                        feature, sheet_id
                    )
                ) from exc
        for subgroup in subgroup_fields:
            values = {
                str(row.get(subgroup, "")).strip() or "<missing>" for row in rows
            }
            record[subgroup] = next(iter(values)) if len(values) == 1 else "<mixed>"
        records.append(record)
    return records


def _parse_binary(value: Any, field: str) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise CalibratorError("{} must contain only binary values".format(field))


def split_sheet_records(
    records: Sequence[Mapping[str, Any]],
    seed: int = 2026,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Stratify unique sheets into disjoint train/calibration/test sets."""

    if train_fraction <= 0 or calibration_fraction < 0:
        raise CalibratorError("Split fractions must be non-negative")
    if train_fraction + calibration_fraction >= 1:
        raise CalibratorError("Train + calibration fractions must be below one")
    if len({record["sheet_id"] for record in records}) != len(records):
        raise CalibratorError("Records must be aggregated to unique sheet_id first")

    randomizer = random.Random(seed)
    by_label: Dict[int, List[Mapping[str, Any]]] = {0: [], 1: []}
    for record in records:
        by_label[int(record["is_error"])].append(record)
    if not by_label[0] or not by_label[1]:
        raise CalibratorError("Dataset must contain correct and erroneous sheets")

    splits = [[], [], []]  # type: ignore
    for label_records in by_label.values():
        randomizer.shuffle(label_records)
        count = len(label_records)
        train_end = max(1, int(round(count * train_fraction)))
        calibration_end = train_end + int(round(count * calibration_fraction))
        calibration_end = min(calibration_end, count)
        splits[0].extend(label_records[:train_end])
        splits[1].extend(label_records[train_end:calibration_end])
        splits[2].extend(label_records[calibration_end:])

    for split in splits:
        randomizer.shuffle(split)
    _assert_disjoint(splits)
    if not splits[0] or not splits[2]:
        raise CalibratorError(
            "Dataset is too small for disjoint train/calibration/test splits"
        )
    return splits[0], splits[1], splits[2]


def _assert_disjoint(splits: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    id_sets = [
        {str(record["sheet_id"]) for record in split} for split in splits
    ]
    for index, left in enumerate(id_sets):
        for right in id_sets[index + 1 :]:
            overlap = left & right
            if overlap:
                raise CalibratorError(
                    "Sheet leakage detected across splits: {}".format(
                        sorted(overlap)[:5]
                    )
                )


def split_hash(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = "\n".join(
        sorted(str(record["sheet_id"]) for record in records)
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_serialized_calibrator(
    train_records: Sequence[Mapping[str, Any]],
    calibration_records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    model_version: str,
    backend: str = "auto",
    calibration: str = "auto",
) -> SerializedRiskCalibrator:
    """Train logistic risk and optionally calibrate on a disjoint split."""

    if backend not in {"auto", "numpy", "sklearn"}:
        raise CalibratorError("backend must be auto, numpy, or sklearn")
    if calibration not in {"auto", "none", "sigmoid", "isotonic"}:
        raise CalibratorError(
            "calibration must be auto, none, sigmoid, or isotonic"
        )
    payload = None
    if backend in {"auto", "sklearn"}:
        try:
            payload = _train_sklearn_logistic(
                train_records, feature_names, model_version
            )
        except ImportError:
            if backend == "sklearn":
                raise CalibratorError(
                    "scikit-learn backend requested but not installed"
                )
    if payload is None:
        payload = train_logistic_model(
            train_records,
            [int(record["is_error"]) for record in train_records],
            feature_names=feature_names,
            model_version=model_version,
        )

    model = SerializedRiskCalibrator(payload)
    if calibration != "none" and calibration_records:
        probabilities = [
            model.predict(record)[0] for record in calibration_records
        ]
        labels = [int(record["is_error"]) for record in calibration_records]
        if len(set(labels)) >= 2 and len(labels) >= 4:
            method = calibration
            if method == "auto":
                method = "isotonic" if len(labels) >= 30 else "sigmoid"
            if method == "isotonic":
                try:
                    payload["post_calibration"] = _fit_isotonic(
                        probabilities, labels
                    )
                except ImportError:
                    if calibration == "isotonic":
                        raise CalibratorError(
                            "isotonic calibration requires scikit-learn"
                        )
                    payload["post_calibration"] = fit_sigmoid_calibration(
                        probabilities, labels
                    )
            else:
                payload["post_calibration"] = fit_sigmoid_calibration(
                    probabilities, labels
                )
    return SerializedRiskCalibrator(payload)


def _train_sklearn_logistic(
    records: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    model_version: str,
) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(
        [[float(record[name]) for name in feature_names] for record in records],
        dtype=np.float64,
    )
    y = np.asarray([int(record["is_error"]) for record in records])
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    estimator = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=0
    )
    estimator.fit((x - means) / scales, y)
    return {
        "format_version": SerializedRiskCalibrator.FORMAT_VERSION,
        "model_type": "logistic_risk",
        "model_version": str(model_version),
        "feature_names": list(feature_names),
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "post_calibration": {"method": "none"},
        "training_backend": "sklearn",
    }


def _fit_isotonic(
    probabilities: Sequence[float], labels: Sequence[int]
) -> Dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression

    estimator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    ).fit(probabilities, labels)
    return {
        "method": "isotonic",
        "x": estimator.X_thresholds_.tolist(),
        "y": estimator.y_thresholds_.tolist(),
    }
