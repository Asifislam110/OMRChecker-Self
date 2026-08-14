"""Build an immutable drift baseline from caller-supplied real records."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.reliable_omr.validation.drift import (
    DEFAULT_CATEGORICAL_FIELDS,
    DEFAULT_NUMERIC_FIELDS,
    build_drift_baseline,
)
from src.reliable_omr.validation.io import read_records, write_json


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize an approved reference window. This command does not "
            "generate, augment, or impute production records."
        )
    )
    parser.add_argument("reference_records", type=Path)
    parser.add_argument("output_baseline", type=Path)
    parser.add_argument("--numeric-field", action="append", default=[])
    parser.add_argument("--categorical-field", action="append", default=[])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-samples", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        records = read_records(args.reference_records)
        baseline = build_drift_baseline(
            records,
            numeric_fields=args.numeric_field or DEFAULT_NUMERIC_FIELDS,
            categorical_fields=(
                args.categorical_field or DEFAULT_CATEGORICAL_FIELDS
            ),
            bin_count=args.bins,
            minimum_samples=args.minimum_samples,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        baseline = {
            "schema_version": 1,
            "status": "input_error",
            "error": str(exc),
        }
        write_json(args.output_baseline, baseline)
        print(json.dumps(baseline, indent=2, sort_keys=True))
        return 1
    write_json(args.output_baseline, baseline)
    print(json.dumps(baseline, indent=2, sort_keys=True))
    return 0 if baseline["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
