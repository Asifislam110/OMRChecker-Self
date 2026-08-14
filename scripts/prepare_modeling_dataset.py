"""Audit and prepare leakage-safe OMR capture/modeling manifests."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
    prepare_capture_dataset,
)
from src.reliable_omr.modeling.readiness import (  # noqa: E402
    classifier_readiness_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate physical OMR captures, hash image bytes, normalize the "
            "CSV, and create deterministic connected-group splits."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument(
        "--bubble-crops",
        type=Path,
        help=(
            "Optional verified bubble-crop CSV used only for lightweight "
            "classifier readiness gates."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        prepared = prepare_capture_dataset(
            args.input_csv,
            args.output_dir,
            seed=args.seed,
            train_fraction=args.train_fraction,
            calibration_fraction=args.calibration_fraction,
        )
        readiness = classifier_readiness_report(
            prepared["normalized_csv"],
            prepared["split_manifest"],
            args.bubble_crops,
        )
    except DatasetContractError as exc:
        raise SystemExit("Dataset audit failed: {}".format(exc))

    readiness_path = args.output_dir / "classifier_readiness.json"
    readiness_path.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "normalized_csv": str(prepared["normalized_csv"]),
        "split_manifest": str(prepared["split_manifest"]),
        "audit_report": str(prepared["audit_report"]),
        "classifier_readiness_report": str(readiness_path),
        "risk_calibrator_training_ready": prepared["manifest"][
            "training_readiness"
        ]["ready"],
        "classifier_benchmark_ready": readiness["benchmark_ready"],
        "note": (
            "Readiness is structural only. No physical-capture accuracy has "
            "been measured."
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
