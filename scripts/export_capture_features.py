"""Export one human-verified capture row from processor diagnostics."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.labeling import (  # noqa: E402
    LabelingContractError,
    export_capture_row,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Join explicit two-reviewer capture truth to "
            "confidence_diagnostics and write one omr-capture-v1 row."
        )
    )
    parser.add_argument("capture_image", type=Path)
    parser.add_argument("processor_result_json", type=Path)
    parser.add_argument("capture_metadata_json", type=Path)
    parser.add_argument("human_verification_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--sheet-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = export_capture_row(
            args.capture_image,
            args.processor_result_json,
            args.capture_metadata_json,
            args.human_verification_json,
            args.output_csv,
            sheet_index=args.sheet_index,
        )
    except LabelingContractError as exc:
        raise SystemExit("Capture export failed: {}".format(exc))
    print(
        json.dumps(
            {
                "capture_csv": str(result["capture_csv"]),
                "capture_id": result["capture_id"],
                "provenance": result["provenance"],
                "note": (
                    "is_error and label_status came only from human "
                    "verification, never from the processor answer."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
