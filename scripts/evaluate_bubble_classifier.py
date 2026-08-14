"""Evaluate a frozen shadow classifier on the untouched test split."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.classifier import (  # noqa: E402
    ClassifierContractError,
    evaluate_bubble_classifier,
)
from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify classifier provenance and report test confusion, "
            "per-class metrics, subgroups, and filled-vs-not risk/coverage."
        )
    )
    parser.add_argument("model_json", type=Path)
    parser.add_argument("normalized_csv", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("bubble_crops_csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = evaluate_bubble_classifier(
            args.model_json,
            args.normalized_csv,
            args.split_manifest,
            args.bubble_crops_csv,
        )
    except (ClassifierContractError, DatasetContractError) as exc:
        raise SystemExit("Classifier evaluation failed: {}".format(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
