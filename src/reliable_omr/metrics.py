"""Evaluation metrics for review-routing risk models."""

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


DEFAULT_COVERAGES = (0.25, 0.5, 0.75, 0.9, 1.0)


def wilson_lower_bound(
    successes: int, total: int, z: float = 1.959963984540054
) -> float:
    """95% Wilson lower confidence bound for a binomial success proportion."""

    if total <= 0:
        return 0.0
    proportion = successes / float(total)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total))
        / total
    )
    return max(0.0, (center - radius) / denominator)


def risk_coverage_metrics(
    risks: Sequence[float],
    is_error: Sequence[int],
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> List[Dict[str, Any]]:
    """Measure accuracy as progressively riskier sheets are auto-accepted."""

    if len(risks) != len(is_error):
        raise ValueError("risks and is_error must have equal lengths")
    if not risks:
        return []
    pairs = [(float(risk), int(label)) for risk, label in zip(risks, is_error)]
    sorted_risks = sorted(risk for risk, _ in pairs)
    total = len(pairs)
    output = []
    for requested_coverage in coverages:
        coverage = float(requested_coverage)
        if not 0 < coverage <= 1:
            raise ValueError("coverage values must be in (0, 1]")
        target_count = min(total, max(1, int(math.ceil(total * coverage))))
        threshold = sorted_risks[target_count - 1]
        # Apply a real threshold and accept all ties. Using the outcome label to
        # break equal-risk ties would make selective accuracy optimistic.
        accepted = [pair for pair in pairs if pair[0] <= threshold]
        accepted_count = len(accepted)
        errors = sum(int(label) for _, label in accepted)
        correct = accepted_count - errors
        output.append(
            {
                "requested_coverage": coverage,
                "actual_coverage": accepted_count / float(total),
                "accepted": accepted_count,
                "errors": errors,
                "accuracy": correct / float(accepted_count),
                "accuracy_wilson_lower_95": wilson_lower_bound(
                    correct, accepted_count
                ),
                "risk_threshold": float(threshold),
            }
        )
    return output


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = np.where(labels == 1)[0]
    negatives = np.where(labels == 0)[0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = scores[positives][:, None] - scores[negatives][None, :]
    return float(
        (np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def _summary(risks: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    risk_array = np.clip(np.asarray(risks, dtype=np.float64), 1e-8, 1 - 1e-8)
    label_array = np.asarray(labels, dtype=np.int64)
    correct = int(np.sum(label_array == 0))
    return {
        "count": int(len(label_array)),
        "error_count": int(np.sum(label_array)),
        "error_rate": float(np.mean(label_array)) if len(label_array) else 0.0,
        "mean_predicted_risk": (
            float(np.mean(risk_array)) if len(risk_array) else 0.0
        ),
        "brier_score": (
            float(np.mean((risk_array - label_array) ** 2))
            if len(label_array)
            else 0.0
        ),
        "log_loss": (
            float(
                -np.mean(
                    label_array * np.log(risk_array)
                    + (1 - label_array) * np.log(1 - risk_array)
                )
            )
            if len(label_array)
            else 0.0
        ),
        "risk_auc": _binary_auc(label_array, risk_array),
        "accuracy_wilson_lower_95": wilson_lower_bound(
            correct, len(label_array)
        ),
    }


def evaluate_risk_records(
    records: Sequence[Mapping[str, Any]],
    risk_field: str = "risk",
    label_field: str = "is_error",
    subgroup_fields: Sequence[str] = (),
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Dict[str, Any]:
    """Evaluate unique sheet records overall, by coverage, and by subgroup."""

    risks = [float(record[risk_field]) for record in records]
    labels = [int(record[label_field]) for record in records]
    result: Dict[str, Any] = {
        "overall": _summary(risks, labels),
        "risk_coverage": risk_coverage_metrics(risks, labels, coverages),
        "subgroups": {},
    }
    for field in subgroup_fields:
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record.get(field, "<missing>")), []).append(
                record
            )
        result["subgroups"][field] = {
            value: _summary(
                [float(record[risk_field]) for record in group],
                [int(record[label_field]) for record in group],
            )
            for value, group in sorted(grouped.items())
        }
    return result
