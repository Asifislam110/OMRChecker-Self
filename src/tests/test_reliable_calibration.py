import csv

from src.reliable_omr.metrics import (
    evaluate_risk_records,
    risk_coverage_metrics,
    wilson_lower_bound,
)
from src.reliable_omr.training import (
    load_sheet_records,
    split_sheet_records,
    train_serialized_calibrator,
)


FEATURES = ["quality_error_count", "ambiguous_fraction"]


def _records():
    rows = []
    for index in range(60):
        is_error = int(index % 2 == 1)
        rows.append(
            {
                "sheet_id": "sheet-{:03d}".format(index),
                "is_error": is_error,
                "quality_error_count": float(is_error),
                "ambiguous_fraction": 0.7 if is_error else 0.02,
                "capture_mode": "mobile" if index % 3 else "scanner",
            }
        )
    return rows


def test_sheet_splits_are_disjoint_and_serialized_model_predicts_risk():
    train, calibration, test = split_sheet_records(_records(), seed=7)
    split_ids = [
        {record["sheet_id"] for record in split}
        for split in (train, calibration, test)
    ]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])

    model = train_serialized_calibrator(
        train,
        calibration,
        FEATURES,
        model_version="test-v1",
        backend="numpy",
        calibration="sigmoid",
    )
    low_risk, _ = model.predict(
        {"quality_error_count": 0, "ambiguous_fraction": 0.01}
    )
    high_risk, _ = model.predict(
        {"quality_error_count": 1, "ambiguous_fraction": 0.8}
    )

    assert model.payload["post_calibration"]["method"] == "sigmoid"
    assert high_risk > low_risk
    assert 0 <= low_risk <= 1
    assert 0 <= high_risk <= 1


def test_csv_rows_are_aggregated_before_sheet_split(tmp_path):
    path = tmp_path / "training.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "sheet_id",
                "is_error",
                "quality_error_count",
                "ambiguous_fraction",
                "capture_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sheet_id": "same-sheet",
                "is_error": 0,
                "quality_error_count": 0,
                "ambiguous_fraction": 0.1,
                "capture_mode": "scanner",
            }
        )
        writer.writerow(
            {
                "sheet_id": "same-sheet",
                "is_error": 1,
                "quality_error_count": 1,
                "ambiguous_fraction": 0.5,
                "capture_mode": "scanner",
            }
        )

    records = load_sheet_records(
        path, FEATURES, subgroup_fields=["capture_mode"]
    )

    assert len(records) == 1
    assert records[0]["source_row_count"] == 2
    assert records[0]["is_error"] == 1
    assert records[0]["quality_error_count"] == 0.5


def test_risk_coverage_wilson_and_subgroups():
    points = risk_coverage_metrics(
        [0.01, 0.05, 0.8, 0.9], [0, 0, 1, 1], coverages=[0.5, 1.0]
    )
    assert points[0]["accepted"] == 2
    assert points[0]["accuracy"] == 1.0
    assert points[1]["errors"] == 2
    assert 0.72 < wilson_lower_bound(90, 100) < 0.84
    tied = risk_coverage_metrics(
        [0.1, 0.1, 0.9], [0, 1, 1], coverages=[0.34]
    )
    assert tied[0]["accepted"] == 2
    assert tied[0]["actual_coverage"] == 2 / 3

    report = evaluate_risk_records(
        [
            {"risk": 0.01, "is_error": 0, "capture_mode": "scanner"},
            {"risk": 0.9, "is_error": 1, "capture_mode": "mobile"},
        ],
        subgroup_fields=["capture_mode"],
        coverages=[1.0],
    )
    assert report["overall"]["risk_auc"] == 1.0
    assert set(report["subgroups"]["capture_mode"]) == {"mobile", "scanner"}
