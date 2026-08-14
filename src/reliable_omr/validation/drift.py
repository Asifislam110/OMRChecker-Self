"""Reference-baseline and production-window monitoring for Reliable OMR."""

import bisect
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.reliable_omr.validation.io import ValidationInputError, parse_binary
from src.reliable_omr.validation.statistics import accuracy_gate


DEFAULT_NUMERIC_FIELDS = ("confidence", "risk", "quality_score")
DEFAULT_CATEGORICAL_FIELDS = (
    "quality_status",
    "status",
    "capture_mode",
    "review_outcome",
    "correction_outcome",
)


class DriftBaselineError(ValueError):
    """Raised when a drift baseline is invalid or has been modified."""


def _canonical_payload(baseline: Mapping[str, Any]) -> bytes:
    payload = dict(baseline)
    payload.pop("baseline_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def baseline_hash(baseline: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(baseline)).hexdigest()


def _optional_number(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _cut_points(values: Sequence[float], bin_count: int) -> List[float]:
    ordered = sorted(values)
    candidates = [
        _quantile(ordered, index / float(bin_count))
        for index in range(1, bin_count)
    ]
    # Quantile-only bins cannot detect a shift outside a constant (or very
    # narrow) reference range because both the reference maximum and larger
    # production values land in the same final bin. Explicit underflow and
    # overflow boundaries keep those shifts observable.
    candidates.extend(
        [
            math.nextafter(ordered[0], -math.inf),
            math.nextafter(ordered[-1], math.inf),
        ]
    )
    return sorted(set(candidates))


def _histogram(values: Sequence[float], cuts: Sequence[float]) -> List[int]:
    counts = [0] * (len(cuts) + 1)
    for value in values:
        counts[bisect.bisect_right(cuts, value)] += 1
    return counts


def _distribution(counts: Sequence[int], missing: int) -> List[float]:
    total = sum(counts) + missing
    if total == 0:
        return [0.0] * (len(counts) + 1)
    return [count / float(total) for count in [*counts, missing]]


def _categorical_counts(
    records: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(field, "")).strip() or "<missing>"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _outcome_summary(
    records: Sequence[Mapping[str, Any]],
    correctness_field: str = "is_correct",
    ground_truth_source_field: str = "ground_truth_source",
    human_source_value: str = "human",
) -> Dict[str, Any]:
    review_statuses = {
        "review",
        "needs_review",
        "in_review",
        "recapture_requested",
    }
    accepted_statuses = {"accepted", "auto_accepted"}
    reviews = sum(
        str(record.get("status", "")).strip().lower() in review_statuses
        for record in records
    )
    correction_observed = 0
    corrections = 0
    for record in records:
        outcome = str(record.get("correction_outcome", "")).strip().lower()
        if outcome in {"corrected", "confirmed_correct", "no_correction"}:
            correction_observed += 1
            corrections += int(outcome == "corrected")

    accepted_human = []
    for record in records:
        if (
            str(record.get("status", "")).strip().lower()
            not in accepted_statuses
        ):
            continue
        if (
            str(record.get(ground_truth_source_field, "")).strip()
            != human_source_value
        ):
            continue
        try:
            accepted_human.append(
                parse_binary(record.get(correctness_field), correctness_field)
            )
        except ValidationInputError:
            continue
    return {
        "reviewed": reviews,
        "review_rate": reviews / float(len(records)) if records else None,
        "correction_outcomes_observed": correction_observed,
        "corrected": corrections,
        "correction_rate_when_observed": (
            corrections / float(correction_observed)
            if correction_observed
            else None
        ),
        "accepted_answers_with_human_ground_truth": len(accepted_human),
        "accepted_human_correct": sum(accepted_human),
    }


def build_drift_baseline(
    records: Sequence[Mapping[str, Any]],
    numeric_fields: Sequence[str] = DEFAULT_NUMERIC_FIELDS,
    categorical_fields: Sequence[str] = DEFAULT_CATEGORICAL_FIELDS,
    bin_count: int = 10,
    minimum_samples: int = 1000,
) -> Dict[str, Any]:
    """Summarize a real reference window.

    This function never synthesizes data.
    """

    if not records:
        raise ValidationInputError("Drift reference input contains no records")
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    numeric = {}
    for field in numeric_fields:
        values = [
            value
            for value in (
                _optional_number(record.get(field)) for record in records
            )
            if value is not None
        ]
        cuts = _cut_points(values, bin_count) if values else []
        counts = _histogram(values, cuts)
        missing = len(records) - len(values)
        numeric[field] = {
            "observed_samples": len(values),
            "missing_samples": missing,
            "cut_points": cuts,
            "bin_counts": counts,
            "probabilities_with_missing_bucket": _distribution(
                counts, missing
            ),
            "mean": (
                sum(values) / float(len(values)) if values else None
            ),
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
        }
    categorical = {}
    for field in categorical_fields:
        counts = _categorical_counts(records, field)
        categorical[field] = {
            "observed_samples": len(records),
            "counts": counts,
            "probabilities": {
                value: count / float(len(records))
                for value, count in counts.items()
            },
        }
    baseline: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "minimum_samples": minimum_samples,
        "status": (
            "ready"
            if len(records) >= minimum_samples
            else "insufficient_data"
        ),
        "numeric_fields": numeric,
        "categorical_fields": categorical,
        "outcomes": _outcome_summary(records),
        "provenance": (
            "Summarized from caller-provided reference records; "
            "no records were generated or imputed."
        ),
    }
    baseline["baseline_sha256"] = baseline_hash(baseline)
    return baseline


def validate_drift_baseline(baseline: Mapping[str, Any]) -> None:
    if int(baseline.get("schema_version", -1)) != 1:
        raise DriftBaselineError("Unsupported drift baseline schema_version")
    expected = str(baseline.get("baseline_sha256", ""))
    if not expected or expected != baseline_hash(baseline):
        raise DriftBaselineError("Drift baseline hash mismatch")
    if not isinstance(baseline.get("numeric_fields"), Mapping):
        raise DriftBaselineError("Baseline numeric_fields must be an object")
    if not isinstance(baseline.get("categorical_fields"), Mapping):
        raise DriftBaselineError(
            "Baseline categorical_fields must be an object"
        )


def _psi(reference: Sequence[float], current: Sequence[float]) -> float:
    epsilon = 1e-6
    return float(
        sum(
            (max(current_value, epsilon) - max(reference_value, epsilon))
            * math.log(
                max(current_value, epsilon) / max(reference_value, epsilon)
            )
            for reference_value, current_value in zip(reference, current)
        )
    )


def _tvd(
    reference: Mapping[str, float], current: Mapping[str, float]
) -> float:
    categories = set(reference).union(current)
    return 0.5 * sum(
        abs(float(reference.get(value, 0.0)) - float(current.get(value, 0.0)))
        for value in categories
    )


def _level(value: float, warning: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    if value >= warning:
        return "warning"
    return "ok"


def monitor_production_window(
    baseline: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    minimum_window_samples: int = 500,
    minimum_field_samples: int = 100,
    minimum_outcome_samples: int = 100,
    psi_warning: float = 0.1,
    psi_alert: float = 0.25,
    tvd_warning: float = 0.1,
    tvd_alert: float = 0.2,
    target_accuracy: float = 0.999,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Compare a production window with an immutable caller baseline."""

    validate_drift_baseline(baseline)
    if not records:
        raise ValidationInputError("Production window contains no records")
    alerts: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {"numeric": {}, "categorical": {}}
    if baseline.get("status") != "ready":
        alerts.append(
            {
                "level": "insufficient_data",
                "code": "baseline_minimum_samples",
                "field": None,
                "observed": int(baseline.get("record_count", 0)),
                "threshold": int(baseline.get("minimum_samples", 1)),
                "message": (
                    "Reference baseline is below its minimum sample count."
                ),
            }
        )
    if len(records) < minimum_window_samples:
        alerts.append(
            {
                "level": "insufficient_data",
                "code": "window_minimum_samples",
                "field": None,
                "observed": len(records),
                "threshold": minimum_window_samples,
                "message": (
                    "Production window is below the minimum sample count."
                ),
            }
        )

    for field, reference in baseline["numeric_fields"].items():
        values = [
            value
            for value in (
                _optional_number(record.get(field)) for record in records
            )
            if value is not None
        ]
        cuts = [float(value) for value in reference["cut_points"]]
        counts = _histogram(values, cuts)
        missing = len(records) - len(values)
        current_probabilities = _distribution(counts, missing)
        reference_probabilities = [
            float(value)
            for value in reference["probabilities_with_missing_bucket"]
        ]
        psi = _psi(reference_probabilities, current_probabilities)
        enough = (
            int(reference["observed_samples"]) >= minimum_field_samples
            and len(values) >= minimum_field_samples
        )
        level = (
            _level(psi, psi_warning, psi_alert)
            if enough
            else "insufficient_data"
        )
        results["numeric"][field] = {
            "level": level,
            "psi": psi,
            "reference_samples": int(reference["observed_samples"]),
            "current_samples": len(values),
            "current_missing": missing,
            "current_bin_counts": counts,
            "cut_points": cuts,
        }
        if level != "ok":
            alerts.append(
                {
                    "level": level,
                    "code": (
                        "numeric_minimum_samples"
                        if level == "insufficient_data"
                        else "numeric_distribution_drift"
                    ),
                    "field": field,
                    "observed": len(values) if not enough else psi,
                    "threshold": (
                        minimum_field_samples
                        if not enough
                        else (psi_alert if level == "alert" else psi_warning)
                    ),
                    "message": (
                        "Numeric telemetry distribution requires attention."
                    ),
                }
            )

    for field, reference in baseline["categorical_fields"].items():
        counts = _categorical_counts(records, field)
        current_probabilities = {
            value: count / float(len(records))
            for value, count in counts.items()
        }
        reference_probabilities = {
            str(value): float(probability)
            for value, probability in reference["probabilities"].items()
        }
        tvd = _tvd(reference_probabilities, current_probabilities)
        enough = (
            int(reference["observed_samples"]) >= minimum_field_samples
            and len(records) >= minimum_field_samples
        )
        level = (
            _level(tvd, tvd_warning, tvd_alert)
            if enough
            else "insufficient_data"
        )
        unseen = sorted(set(counts) - set(reference_probabilities))
        results["categorical"][field] = {
            "level": level,
            "total_variation_distance": tvd,
            "current_counts": counts,
            "unseen_categories": unseen,
        }
        if level != "ok" or unseen:
            alert_level = "alert" if unseen and level == "ok" else level
            alerts.append(
                {
                    "level": alert_level,
                    "code": (
                        "unseen_category"
                        if unseen
                        else (
                            "categorical_minimum_samples"
                            if level == "insufficient_data"
                            else "categorical_distribution_drift"
                        )
                    ),
                    "field": field,
                    "observed": unseen if unseen else tvd,
                    "threshold": (
                        minimum_field_samples
                        if not enough
                        else (tvd_alert if level == "alert" else tvd_warning)
                    ),
                    "message": (
                        "Categorical telemetry distribution requires "
                        "attention."
                    ),
                }
            )

    outcomes = _outcome_summary(records)
    accepted_human_gate = accuracy_gate(
        int(outcomes["accepted_human_correct"]),
        int(outcomes["accepted_answers_with_human_ground_truth"]),
        target_accuracy=target_accuracy,
        confidence_level=confidence_level,
        minimum_samples=minimum_outcome_samples,
    )
    outcomes["accepted_human_accuracy_gate"] = accepted_human_gate
    if (
        int(outcomes["correction_outcomes_observed"])
        < minimum_outcome_samples
    ):
        alerts.append(
            {
                "level": "insufficient_data",
                "code": "correction_outcome_minimum_samples",
                "field": "correction_outcome",
                "observed": outcomes["correction_outcomes_observed"],
                "threshold": minimum_outcome_samples,
                "message": (
                    "Too few correction outcomes to assess outcome stability."
                ),
            }
        )
    if accepted_human_gate["status"] != "pass":
        alerts.append(
            {
                "level": (
                    "alert"
                    if accepted_human_gate["status"] == "fail"
                    else "insufficient_data"
                ),
                "code": "accepted_human_accuracy",
                "field": "is_correct",
                "observed": accepted_human_gate["one_sided_lower_bound"],
                "threshold": target_accuracy,
                "message": (
                    "Human-ground-truthed accepted accuracy is not "
                    "established."
                ),
            }
        )

    levels = {str(alert["level"]) for alert in alerts}
    if "alert" in levels:
        status = "alert"
    elif "insufficient_data" in levels:
        status = "insufficient_data"
    elif "warning" in levels:
        status = "warning"
    else:
        status = "ok"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "baseline_sha256": baseline["baseline_sha256"],
        "window_samples": len(records),
        "minimums": {
            "window_samples": minimum_window_samples,
            "field_samples": minimum_field_samples,
            "outcome_samples": minimum_outcome_samples,
        },
        "distributions": results,
        "outcomes": outcomes,
        "baseline_outcomes": baseline.get("outcomes", {}),
        "alerts": alerts,
        "limitations": [
            (
                "Production correctness is measurable only where independent "
                "human ground truth or correction outcomes are captured."
            ),
            (
                "Drift alerts diagnose distribution change; they do not prove "
                "a causal accuracy regression."
            ),
        ],
    }
