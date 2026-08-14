import copy
import json
from pathlib import Path

import pytest

from scripts.validate_reliable_omr_accuracy import main as validate_main
from src.reliable_omr.validation.drift import (
    build_drift_baseline,
    monitor_production_window,
)
from src.reliable_omr.validation.io import read_records
from src.reliable_omr.validation.manifests import (
    ManifestError,
    create_split_manifest,
    manifest_hash,
    partition_records,
    validate_split_manifest,
)
from src.reliable_omr.validation.release import build_release_report
from src.reliable_omr.validation.statistics import (
    accuracy_gate,
    clopper_pearson_lower_bound,
    zero_error_sample_requirement,
)


FIXTURES = Path(__file__).parent / "fixtures" / "reliable_omr_validation"


def test_exact_one_sided_gate_requires_enough_zero_error_samples():
    assert zero_error_sample_requirement(0.999, 0.95) == 2995
    insufficient = accuracy_gate(2994, 2994, 0.999, 0.95)
    passing = accuracy_gate(2995, 2995, 0.999, 0.95)

    assert insufficient["point_accuracy"] == 1.0
    assert insufficient["status"] == "insufficient_data"
    assert passing["status"] == "pass"
    assert passing["one_sided_lower_bound"] >= 0.999
    assert 0.82 < clopper_pearson_lower_bound(90, 100, 0.95) < 0.85


def test_manifest_detects_tampering_overlap_and_omitted_groups():
    manifest = create_split_manifest(
        ["group-{}".format(index) for index in range(12)],
        calibration_fraction=0.25,
        test_fraction=0.25,
    )
    validate_split_manifest(manifest)

    tampered = copy.deepcopy(manifest)
    tampered["splits"]["test"].append("new-group")
    with pytest.raises(ManifestError, match="hash mismatch"):
        validate_split_manifest(tampered)

    overlap = copy.deepcopy(manifest)
    overlap["splits"]["test"][0] = overlap["splits"]["train"][0]
    overlap["splits"]["test"].sort()
    overlap["manifest_sha256"] = manifest_hash(overlap)
    with pytest.raises(ManifestError, match="leakage"):
        validate_split_manifest(overlap)

    represented = [
        {"group_id": group_id}
        for group_id in manifest["splits"]["calibration"]
        + manifest["splits"]["test"]
    ]
    with pytest.raises(ManifestError, match="omits manifest groups"):
        partition_records(represented, manifest)


def test_manifest_rejects_record_level_cherry_picking_within_groups():
    group_ids = [
        "group-{}".format(index)
        for index in range(6)
        for _ in range(2)
    ]
    manifest = create_split_manifest(group_ids)
    records = [{"group_id": group_id} for group_id in group_ids]
    records.pop()

    with pytest.raises(ManifestError, match="record-level cherry-picking"):
        partition_records(records, manifest)


def _release_records(manifest, test_risks=(0.1, 0.2, 0.3, 0.4)):
    records = []
    for split_name, group_ids in manifest["splits"].items():
        for group_id in group_ids:
            risks = (
                test_risks
                if split_name == "test"
                else (0.1, 0.2, 0.3, 0.4)
            )
            modes = ("scanner", "mobile", "scanner", "mobile")
            for answer_index, (risk, mode) in enumerate(zip(risks, modes)):
                records.append(
                    {
                        "answer_id": "{}-{}".format(group_id, answer_index),
                        "group_id": group_id,
                        "risk": risk,
                        "is_correct": 1,
                        "ground_truth_source": "human",
                        "capture_mode": mode,
                    }
                )
    return records


def test_threshold_is_selected_on_calibration_and_test_is_gated_once():
    manifest = create_split_manifest(
        [
            "group-{:02d}".format(index)
            for index in range(30)
            for _ in range(4)
        ],
        calibration_fraction=0.3,
        test_fraction=0.3,
    )
    common = {
        "required_subgroups": {
            "capture_mode": ["scanner", "mobile"],
        },
        "target_accuracy": 0.5,
        "confidence_level": 0.8,
        "minimum_accepted": 5,
        "subgroup_minimum_accepted": 5,
        "coverages": [1.0],
    }
    report = build_release_report(
        _release_records(manifest), manifest, **common)
    changed_test = build_release_report(
        _release_records(manifest, test_risks=(0.99, 0.99, 0.99, 0.99)),
        manifest,
        **common
    )

    assert report["passed"] is True
    selection = report["threshold_selection"]
    assert selection["test_records_used_for_selection"] is False
    assert report["threshold_selection"]["selected_threshold"] == 0.4
    assert changed_test["threshold_selection"]["selected_threshold"] == 0.4
    assert changed_test["final_test"]["status"] == "insufficient_data"
    assert set(report["final_test"]["subgroup_gates"]["capture_mode"]) == {
        "mobile",
        "scanner",
    }


def _monitor_record(index, shifted=False):
    corrected = index % 10 == 0
    return {
        "confidence": 0.1 if shifted else 0.5,
        "risk": 0.9 if shifted else 0.5,
        "quality_score": 0.2 if shifted else 0.5,
        "quality_status": "poor" if shifted else "good",
        "status": "needs_review" if shifted else "auto_accepted",
        "capture_mode": "mobile" if shifted else "scanner",
        "review_outcome": "corrected" if corrected else "confirmed",
        "correction_outcome": (
            "corrected" if corrected else "confirmed_correct"
        ),
        "is_correct": int(
            not corrected),
        "ground_truth_source": "human",
    }


def test_drift_monitor_alerts_on_shift_and_small_windows():
    baseline_records = [_monitor_record(index) for index in range(200)]
    baseline = build_drift_baseline(
        baseline_records, minimum_samples=100, bin_count=5
    )
    shifted = [_monitor_record(index, shifted=True) for index in range(200)]
    report = monitor_production_window(
        baseline,
        shifted,
        minimum_window_samples=100,
        minimum_field_samples=50,
        minimum_outcome_samples=20,
        target_accuracy=0.8,
        confidence_level=0.8,
    )
    small = monitor_production_window(
        baseline,
        baseline_records[:5],
        minimum_window_samples=10,
        minimum_field_samples=1,
        minimum_outcome_samples=1,
        target_accuracy=0.5,
        confidence_level=0.8,
    )

    assert report["status"] == "alert"
    assert report["distributions"]["numeric"]["risk"]["level"] == "alert"
    assert any(alert["code"] ==
               "unseen_category" for alert in report["alerts"])
    assert small["status"] == "insufficient_data"


def test_accuracy_cli_writes_json_and_fails_on_synthetic_small_sample(
        tmp_path):
    records_path = FIXTURES / "synthetic_benchmark.jsonl"
    records = read_records(records_path)
    manifest = create_split_manifest(
        [record["group_id"] for record in records],
        calibration_fraction=0.33,
        test_fraction=0.33,
    )
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = validate_main(
        [
            str(records_path),
            str(manifest_path),
            "--output",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert report["passed"] is False
    assert report["decision"] == "insufficient_data"
    assert report["final_test"]["status"] == "not_evaluated"
