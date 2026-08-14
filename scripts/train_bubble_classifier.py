"""Train a gated shadow-only HOG/logistic bubble classifier."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.classifier import (  # noqa: E402
    ClassifierContractError,
    train_bubble_classifier,
)
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Require the classifier readiness gate, select logistic C on the "
            "calibration split, evaluate once on test, and export numeric "
            "JSON."
        )
    )
    parser.add_argument("normalized_csv", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("bubble_crops_csv", type=Path)
    parser.add_argument("output_model", type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--c",
        type=float,
        action="append",
        dest="c_values",
        help="Repeat to replace the default C candidates 0.1, 1, and 10.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = train_bubble_classifier(
            args.normalized_csv,
            args.split_manifest,
            args.bubble_crops_csv,
            args.output_model,
            model_version=args.model_version,
            seed=args.seed,
            c_values=args.c_values or (0.1, 1.0, 10.0),
        )
    except (ClassifierContractError, DatasetContractError) as exc:
        raise SystemExit("Classifier training failed: {}".format(exc))
    print(
        json.dumps(
            {
                "model_path": str(result["model_path"]),
                "deployment_status": "shadow_only",
                "metrics": result["metrics"],
                "note": (
                    "This benchmark export is not enabled in the recognizer "
                    "or service."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
