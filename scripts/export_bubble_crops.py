"""Export canonical bubble crops from processor geometry and human labels."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.modeling.labeling import (  # noqa: E402
    LabelingContractError,
    export_bubble_crops,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate two-reviewer bubble annotations, crop the rectified "
            "image, and write an omr-bubble-crop-v1 CSV."
        )
    )
    parser.add_argument("rectified_image", type=Path)
    parser.add_argument("processor_result_json", type=Path)
    parser.add_argument("human_annotations_csv", type=Path)
    parser.add_argument("capture_metadata_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sheet-index", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--padding-ratio", type=float, default=1.65)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = export_bubble_crops(
            args.rectified_image,
            args.processor_result_json,
            args.human_annotations_csv,
            args.capture_metadata_json,
            args.output_dir,
            sheet_index=args.sheet_index,
            crop_size=args.crop_size,
            padding_ratio=args.padding_ratio,
        )
    except LabelingContractError as exc:
        raise SystemExit("Bubble export failed: {}".format(exc))
    print(
        json.dumps(
            {
                "bubble_csv": str(result["bubble_csv"]),
                "crops_dir": str(result["crops_dir"]),
                "crop_count": result["crop_count"],
                "provenance": result["provenance"],
                "note": (
                    "Labels came only from the verified/adjudicated human "
                    "CSV; machine answers were not used as truth."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
