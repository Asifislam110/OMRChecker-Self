"""Leakage-safe threshold selection and exact accuracy release gates."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.reliable_omr.metrics import risk_coverage_metrics
from src.reliable_omr.validation.io import (
    ValidationInputError,
    finite_float,
    parse_binary,
)
from src.reliable_omr.validation.manifests import (
    manifest_hash,
    partition_records,
)
from src.reliable_omr.validation.statistics import accuracy_gate


DEFAULT_COVERAGES = (0.25, 0.5, 0.75, 0.9, 1.0)


def _value(record: Mapping[str, Any], field: str) -> str:
    value = str(record.get(field, "")).strip()
    return value or "<missing>"


def _normalize_records(
    records: Sequence[Mapping[str, Any]],
    risk_field: str,
    correct_field: str,
    answer_id_field: str,
    ground_truth_source_field: str,
    ground_truth_source_value: str,
) -> List[Dict[str, Any]]:
    output = []
    seen_answer_ids = set()
    for index, source in enumerate(records):
        record = dict(source)
        if answer_id_field not in record:
            raise ValidationInputError(
                "Record {} is missing answer ID field '{}'".format(
                    index, answer_id_field
                )
            )
        answer_id = str(record[answer_id_field]).strip()
        if not answer_id:
            raise ValidationInputError("Every answer ID must be non-empty")
        if answer_id in seen_answer_ids:
            raise ValidationInputError(
                "Duplicate answer ID '{}' would double-count evidence".format(
                    answer_id
                )
            )
        seen_answer_ids.add(answer_id)

        if risk_field not in record or correct_field not in record:
            raise ValidationInputError(
                "Record '{}' is missing risk or correctness".format(answer_id)
            )
        risk = finite_float(record[risk_field], risk_field)
        if not 0.0 <= risk <= 1.0:
            raise ValidationInputError(
                "{} must be in [0, 1]".format(risk_field))
        source_name = str(record.get(ground_truth_source_field, "")).strip()
        if source_name != ground_truth_source_value:
            raise ValidationInputError(
                (
                    "Record '{}' lacks required human ground truth "
                    "source '{}'"
                ).format(answer_id, ground_truth_source_value)
            )
        record["_validation_risk"] = risk
        record["_validation_correct"] = parse_binary(
            record[correct_field], correct_field
        )
        output.append(record)
    return output


def _gate_status(gates: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(gate["status"]) for gate in gates}
    if "fail" in statuses:
        return "fail"
    if "insufficient_data" in statuses:
        return "insufficient_data"
    return "pass"


def _subgroup_values(
    records: Sequence[Mapping[str, Any]],
    subgroup_fields: Sequence[str],
    required_subgroups: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    output = {}
    for field in subgroup_fields:
        observed = {_value(record, field) for record in records}
        observed.update(str(value)
                        for value in required_subgroups.get(field, ()))
        output[field] = sorted(observed)
    return output


def _gate_counts(
    correct: int,
    total: int,
    target_accuracy: float,
    confidence_level: float,
    minimum_samples: int,
) -> Dict[str, Any]:
    return accuracy_gate(
        correct,
        total,
        target_accuracy=target_accuracy,
        confidence_level=confidence_level,
        minimum_samples=minimum_samples,
    )


def select_risk_threshold(
    calibration_records: Sequence[Mapping[str, Any]],
    subgroup_fields: Sequence[str],
    required_subgroups: Mapping[str, Sequence[str]],
    target_accuracy: float,
    confidence_level: float,
    minimum_accepted: int,
    subgroup_minimum_accepted: int,
) -> Dict[str, Any]:
    """Select maximum passing coverage using only calibration records."""

    if not calibration_records:
        raise ValidationInputError(
            "Calibration split contains no answer records")
    subgroup_values = _subgroup_values(
        calibration_records, subgroup_fields, required_subgroups
    )
    ordered = sorted(
        calibration_records, key=lambda record: float(
            record["_validation_risk"])
    )
    total_correct = 0
    total_accepted = 0
    subgroup_counts: Dict[Tuple[str, str], List[int]] = {
        (field, value): [0, 0]
        for field, values in subgroup_values.items()
        for value in values
    }
    selected: Optional[Dict[str, Any]] = None
    candidates_considered = 0
    decisive_failure_seen = False
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index]["_validation_risk"])
        tied_end = index
        while (
            tied_end < len(ordered)
            and float(ordered[tied_end]["_validation_risk"]) == threshold
        ):
            record = ordered[tied_end]
            correct = int(record["_validation_correct"])
            total_correct += correct
            total_accepted += 1
            for field in subgroup_fields:
                counts = subgroup_counts[(field, _value(record, field))]
                counts[0] += correct
                counts[1] += 1
            tied_end += 1

        overall = _gate_counts(
            total_correct,
            total_accepted,
            target_accuracy,
            confidence_level,
            minimum_accepted,
        )
        subgroups = {}
        flat_gates = [overall]
        for field, values in subgroup_values.items():
            field_gates = {}
            for value in values:
                correct, accepted = subgroup_counts[(field, value)]
                gate = _gate_counts(
                    correct,
                    accepted,
                    target_accuracy,
                    confidence_level,
                    subgroup_minimum_accepted,
                )
                field_gates[value] = gate
                flat_gates.append(gate)
            subgroups[field] = field_gates
        status = _gate_status(flat_gates)
        if status == "pass":
            selected = {
                "risk_threshold": threshold,
                "accepted": total_accepted,
                "coverage": total_accepted / float(len(ordered)),
                "overall_gate": overall,
                "subgroup_gates": subgroups,
            }
        elif status == "fail" and all(
            gate["status"] != "insufficient_data" for gate in flat_gates
        ):
            decisive_failure_seen = True
        candidates_considered += 1
        index = tied_end

    if selected is None:
        status = "fail" if decisive_failure_seen else "insufficient_data"
        reason = (
            "No calibration-only threshold satisfies every accuracy gate."
            if status == "fail"
            else "Calibration data is insufficient for a passing threshold."
        )
        return {
            "status": status,
            "reason": reason,
            "selected_threshold": None,
            "candidates_considered": candidates_considered,
            "selection_split": "calibration",
            "test_records_used_for_selection": False,
        }
    return {
        "status": "pass",
        "reason": "Maximum-coverage threshold passing all calibration gates.",
        "selected_threshold": selected["risk_threshold"],
        "selected_candidate": selected,
        "candidates_considered": candidates_considered,
        "selection_split": "calibration",
        "test_records_used_for_selection": False,
    }


def _evaluate_threshold(
    records: Sequence[Mapping[str, Any]],
    threshold: float,
    subgroup_fields: Sequence[str],
    required_subgroups: Mapping[str, Sequence[str]],
    target_accuracy: float,
    confidence_level: float,
    minimum_accepted: int,
    subgroup_minimum_accepted: int,
) -> Dict[str, Any]:
    accepted = [
        record
        for record in records
        if float(record["_validation_risk"]) <= threshold
    ]
    overall = _gate_counts(
        sum(int(record["_validation_correct"]) for record in accepted),
        len(accepted),
        target_accuracy,
        confidence_level,
        minimum_accepted,
    )
    subgroup_values = _subgroup_values(
        records, subgroup_fields, required_subgroups
    )
    subgroups = {}
    flat_gates = [overall]
    for field, values in subgroup_values.items():
        field_reports = {}
        for value in values:
            all_in_group = [
                record for record in records if _value(record, field) == value
            ]
            accepted_in_group = [
                record
                for record in all_in_group
                if float(record["_validation_risk"]) <= threshold
            ]
            gate = _gate_counts(
                sum(
                    int(record["_validation_correct"])
                    for record in accepted_in_group
                ),
                len(accepted_in_group),
                target_accuracy,
                confidence_level,
                subgroup_minimum_accepted,
            )
            gate["total_answers"] = len(all_in_group)
            gate["reviewed_answers"] = len(
                all_in_group) - len(accepted_in_group)
            field_reports[value] = gate
            flat_gates.append(gate)
        subgroups[field] = field_reports
    status = _gate_status(flat_gates)
    return {
        "status": status,
        "passed": status == "pass",
        "risk_threshold": threshold,
        "routing": {
            "total_answers": len(records),
            "auto_accepted_answers": len(accepted),
            "review_answers": len(records) -
            len(accepted),
            "coverage": len(accepted) /
            float(
                len(records)) if records else 0.0,
            "policy": (
                "risk <= threshold is auto-accepted; all others require review"
            ),
        },
        "overall_gate": overall,
        "subgroup_gates": subgroups,
    }


def _risk_coverage_report(
    records: Sequence[Mapping[str, Any]],
    coverages: Sequence[float],
    target_accuracy: float,
    confidence_level: float,
) -> List[Dict[str, Any]]:
    points = risk_coverage_metrics(
        [float(record["_validation_risk"]) for record in records],
        [1 - int(record["_validation_correct"]) for record in records],
        coverages=coverages,
    )
    for point in points:
        exact_gate = accuracy_gate(
            int(point["accepted"]) - int(point["errors"]),
            int(point["accepted"]),
            target_accuracy=target_accuracy,
            confidence_level=confidence_level,
            minimum_samples=1,
        )
        point["exact_one_sided_accuracy_gate"] = exact_gate
    return points


def build_release_report(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    risk_field: str = "risk",
    correct_field: str = "is_correct",
    answer_id_field: str = "answer_id",
    ground_truth_source_field: str = "ground_truth_source",
    ground_truth_source_value: str = "human",
    subgroup_fields: Sequence[str] = ("capture_mode",),
    required_subgroups: Optional[Mapping[str, Sequence[str]]] = None,
    target_accuracy: float = 0.999,
    confidence_level: float = 0.95,
    minimum_accepted: int = 3000,
    subgroup_minimum_accepted: int = 3000,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> Dict[str, Any]:
    """Build the release decision while keeping final test out of selection."""

    required = dict(required_subgroups or {})
    normalized = _normalize_records(
        records,
        risk_field,
        correct_field,
        answer_id_field,
        ground_truth_source_field,
        ground_truth_source_value,
    )
    partitions = partition_records(normalized, manifest)
    threshold_selection = select_risk_threshold(
        partitions["calibration"],
        subgroup_fields,
        required,
        target_accuracy,
        confidence_level,
        minimum_accepted,
        subgroup_minimum_accepted,
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": threshold_selection["status"],
        "passed": False,
        "target": {
            "auto_accepted_accuracy": target_accuracy,
            "confidence_level": confidence_level,
            "minimum_accepted": minimum_accepted,
            "subgroup_minimum_accepted": subgroup_minimum_accepted,
        },
        "statistical_method": (
            "One-sided exact Clopper-Pearson lower binomial confidence bound"
        ),
        "manifest": {
            "manifest_sha256": manifest_hash(manifest),
            "group_id_field": manifest["group_id_field"],
            "split_group_counts": {
                name: len(values)
                for name, values in manifest["splits"].items()
            },
        },
        "record_counts": {
            name: len(values) for name, values in partitions.items()
        },
        "threshold_selection": threshold_selection,
        "calibration_risk_coverage": _risk_coverage_report(
            partitions["calibration"],
            coverages,
            target_accuracy,
            confidence_level,
        ),
        "limitations": [
            (
                "Correctness is measurable only for answers with independent "
                "human ground truth."
            ),
            (
                "Zero observed errors is not sufficient unless the accepted "
                "sample count supports the configured confidence bound."
            ),
            (
                "A passing benchmark does not eliminate deployment drift; use "
                "the production-window monitor."
            ),
        ],
    }
    threshold = threshold_selection.get("selected_threshold")
    if threshold is None:
        report["final_test"] = {
            "status": "not_evaluated",
            "reason": (
                "No threshold passed calibration; the final test was left "
                "untouched."
            ),
        }
        return report

    final_test = _evaluate_threshold(
        partitions["test"],
        float(threshold),
        subgroup_fields,
        required,
        target_accuracy,
        confidence_level,
        minimum_accepted,
        subgroup_minimum_accepted,
    )
    final_test["risk_coverage"] = _risk_coverage_report(
        partitions["test"],
        coverages,
        target_accuracy,
        confidence_level,
    )
    report["final_test"] = final_test
    report["decision"] = final_test["status"]
    report["passed"] = bool(final_test["passed"])
    return report
