"""Create a content-addressed group split manifest for OMR validation."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.reliable_omr.validation.io import read_records, write_json
from src.reliable_omr.validation.manifests import create_split_manifest


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Assign whole leakage groups to immutable train/calibration/test "
            "splits. Commit and reuse the resulting manifest."
        )
    )
    parser.add_argument("input_records", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--group-id-field", default="group_id")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = read_records(args.input_records)
    group_ids = []
    for index, record in enumerate(records):
        if args.group_id_field not in record:
            raise ValueError(
                "Record {} is missing group field '{}'".format(
                    index, args.group_id_field
                )
            )
        group_ids.append(record[args.group_id_field])
    manifest = create_split_manifest(
        group_ids,
        group_id_field=args.group_id_field,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        test_fraction=args.test_fraction,
    )
    write_json(args.output_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
