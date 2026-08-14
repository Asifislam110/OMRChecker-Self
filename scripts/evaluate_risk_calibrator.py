"""Evaluate a serialized OMR risk calibrator on unique held-out sheets."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.calibration import (  # noqa: E402
    SerializedRiskCalibrator,
)
from src.reliable_omr.metrics import evaluate_risk_records  # noqa: E402
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
    load_prepared_splits,
    sha256_file,
)
from src.reliable_omr.training import load_sheet_records  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split",
        choices=["train", "calibration", "test", "all"],
        default="test",
    )
    parser.add_argument("--sheet-id-field", default="sheet_id")
    parser.add_argument("--label-field", default="is_error")
    parser.add_argument("--subgroup", action="append", default=[])
    parser.add_argument(
        "--coverages", default="0.25,0.5,0.75,0.9,1.0"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    model = SerializedRiskCalibrator.load(args.model)
    if args.split_manifest:
        subgroup_fields = list(
            dict.fromkeys(["capture_mode", *args.subgroup])
        )
        try:
            prepared, manifest = load_prepared_splits(
                args.input_csv,
                args.split_manifest,
                feature_names=model.payload["feature_names"],
                subgroup_fields=subgroup_fields,
            )
        except DatasetContractError as exc:
            raise SystemExit(
                "Prepared dataset verification failed: {}".format(exc)
            )
        if args.split == "all":
            records = [
                record
                for split_records in prepared.values()
                for record in split_records
            ]
        else:
            records = prepared[args.split]
        provenance = {
            "dataset_schema_version": manifest["dataset_schema_version"],
            "normalized_csv_sha256": manifest["provenance"][
                "normalized_csv_sha256"
            ],
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "evaluated_split": args.split,
        }
    else:
        records = load_sheet_records(
            args.input_csv,
            model.payload["feature_names"],
            sheet_id_field=args.sheet_id_field,
            label_field=args.label_field,
            subgroup_fields=args.subgroup,
        )
        subgroup_fields = args.subgroup
        provenance = {"data_contract": "legacy-sheet-csv"}
    evaluated = []
    for record in records:
        copy = dict(record)
        copy["risk"] = model.predict(record)[0]
        evaluated.append(copy)
    coverages = [
        float(item) for item in args.coverages.split(",") if item.strip()
    ]
    report = {
        "model_version": model.payload["model_version"],
        "sheet_count": len(evaluated),
        "provenance": provenance,
        "metrics": evaluate_risk_records(
            evaluated,
            subgroup_fields=subgroup_fields,
            coverages=coverages,
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
