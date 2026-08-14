"""Train the existing JSON risk calibrator from verified group-safe splits."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.calibration import RISK_FEATURES  # noqa: E402
from src.reliable_omr.metrics import evaluate_risk_records  # noqa: E402
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
    load_prepared_splits,
    sha256_file,
)
from src.reliable_omr.training import (  # noqa: E402
    load_sheet_records,
    split_hash,
    split_sheet_records,
    train_serialized_calibrator,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train logistic OMR review risk. Prepared input verifies physical "
            "capture-group splits; legacy rows are aggregated by sheet."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Prepared splits.json. When supplied, SHA-256 provenance and all "
            "physical/session/device/print-batch leakage groups are verified."
        ),
    )
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--sheet-id-field", default="sheet_id")
    parser.add_argument("--label-field", default="is_error")
    parser.add_argument("--features", default=",".join(RISK_FEATURES))
    parser.add_argument("--subgroup", action="append", default=[])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--backend", choices=["auto", "numpy", "sklearn"], default="auto"
    )
    parser.add_argument(
        "--calibration",
        choices=["auto", "none", "sigmoid", "isotonic"],
        default="auto",
    )
    parser.add_argument("--metrics-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    features = [
        item.strip() for item in args.features.split(",") if item.strip()
    ]
    if args.split_manifest:
        prepared_subgroups = list(
            dict.fromkeys(["capture_mode", *args.subgroup])
        )
        try:
            prepared, manifest = load_prepared_splits(
                args.input_csv,
                args.split_manifest,
                feature_names=features,
                subgroup_fields=prepared_subgroups,
                require_training_ready=True,
            )
        except DatasetContractError as exc:
            raise SystemExit(
                "Prepared dataset verification failed: {}".format(exc)
            )
        train = prepared["train"]
        calibration = prepared["calibration"]
        test = prepared["test"]
        subgroup_fields = prepared_subgroups
        training_metadata = {
            "unit": "capture",
            "dataset_schema_version": manifest["dataset_schema_version"],
            "seed": manifest["seed"],
            "group_fields": manifest["group_fields"],
            "normalized_csv_sha256": manifest["provenance"][
                "normalized_csv_sha256"
            ],
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "split_assignment_sha256": manifest["provenance"][
                "split_assignment_sha256"
            ],
            "test_set_used_for_fitting": False,
        }
        data_contract = manifest["manifest_version"]
    else:
        records = load_sheet_records(
            args.input_csv,
            features,
            sheet_id_field=args.sheet_id_field,
            label_field=args.label_field,
            subgroup_fields=args.subgroup,
        )
        train, calibration, test = split_sheet_records(records, seed=args.seed)
        subgroup_fields = args.subgroup
        training_metadata = {
            "unit": "sheet",
            "seed": args.seed,
            "test_set_used_for_fitting": False,
        }
        data_contract = "legacy-sheet-csv"
    model = train_serialized_calibrator(
        train,
        calibration,
        features,
        model_version=args.model_version,
        backend=args.backend,
        calibration=args.calibration,
    )
    training_metadata.update(
        {
            "train_count": len(train),
            "calibration_count": len(calibration),
            "test_count": len(test),
            "train_sheet_id_sha256": split_hash(train),
            "calibration_sheet_id_sha256": split_hash(calibration),
            "test_sheet_id_sha256": split_hash(test),
        }
    )
    model.payload["training_metadata"] = training_metadata
    model.save(args.output_model)

    evaluated = []
    for record in test:
        copy = dict(record)
        copy["risk"] = model.predict(record)[0]
        evaluated.append(copy)
    metrics = evaluate_risk_records(
        evaluated, subgroup_fields=subgroup_fields
    )
    summary = {
        "data_contract": data_contract,
        "model_path": str(args.output_model),
        "model_version": args.model_version,
        "post_calibration": model.payload["post_calibration"]["method"],
        "split_counts": {
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
        },
        "test_metrics": metrics,
    }
    if args.metrics_output:
        args.metrics_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
