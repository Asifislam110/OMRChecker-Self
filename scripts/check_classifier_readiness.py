"""Gate a lightweight bubble-classifier benchmark on real verified data."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.dataset import (  # noqa: E402
    DatasetContractError,
)
from src.reliable_omr.modeling.readiness import (  # noqa: E402
    classifier_readiness_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate bubble crops against capture-level splits and report "
            "whether a lightweight benchmark is justified."
        )
    )
    parser.add_argument("normalized_csv", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("bubble_crops_csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = classifier_readiness_report(
            args.normalized_csv,
            args.split_manifest,
            args.bubble_crops_csv,
        )
    except DatasetContractError as exc:
        raise SystemExit("Classifier readiness audit failed: {}".format(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
