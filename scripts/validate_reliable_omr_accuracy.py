"""Run the conservative Reliable OMR accuracy release gate."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.reliable_omr.validation.io import (
    read_records,
    unique_strings,
    write_json,
)
from src.reliable_omr.validation.manifests import load_split_manifest
from src.reliable_omr.validation.release import build_release_report


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Select an auto-accept risk threshold on calibration groups only, "
            "then gate the untouched test groups with exact confidence bounds."
        )
    )
    parser.add_argument("input_records", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--risk-field", default="risk")
    parser.add_argument("--correct-field", default="is_correct")
    parser.add_argument("--answer-id-field", default="answer_id")
    parser.add_argument(
        "--ground-truth-source-field", default="ground_truth_source"
    )
    parser.add_argument("--ground-truth-source-value", default="human")
    parser.add_argument("--subgroup-field", action="append", default=[])
    parser.add_argument(
        "--required-subgroup",
        action="append",
        default=["capture_mode=scanner", "capture_mode=mobile"],
        help="Required gate as FIELD=VALUE; repeat for additional subgroups.",
    )
    parser.add_argument("--target-accuracy", type=float, default=0.999)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-accepted", type=int, default=3000)
    parser.add_argument(
        "--subgroup-minimum-accepted", type=int, default=3000
    )
    parser.add_argument(
        "--coverages", default="0.25,0.5,0.75,0.9,1.0"
    )
    return parser.parse_args(argv)


def _required_subgroups(values: Sequence[str]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--required-subgroup must use FIELD=VALUE")
        field, value = (part.strip() for part in item.split("=", 1))
        if not field or not value:
            raise ValueError(
                "--required-subgroup must use non-empty FIELD=VALUE")
        output.setdefault(field, []).append(value)
    return output


def _render(payload, output: Optional[Path]) -> None:
    if output:
        write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        required = _required_subgroups(args.required_subgroup)
        subgroup_fields = unique_strings(
            [*args.subgroup_field, *required.keys()]
        )
        records = read_records(args.input_records)
        manifest = load_split_manifest(args.split_manifest)
        coverages = [
            float(value)
            for value in args.coverages.split(",")
            if value.strip()
        ]
        report = build_release_report(
            records,
            manifest,
            risk_field=args.risk_field,
            correct_field=args.correct_field,
            answer_id_field=args.answer_id_field,
            ground_truth_source_field=args.ground_truth_source_field,
            ground_truth_source_value=args.ground_truth_source_value,
            subgroup_fields=subgroup_fields,
            required_subgroups=required,
            target_accuracy=args.target_accuracy,
            confidence_level=args.confidence_level,
            minimum_accepted=args.minimum_accepted,
            subgroup_minimum_accepted=args.subgroup_minimum_accepted,
            coverages=coverages,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "decision": "input_error",
            "passed": False,
            "error": str(exc),
        }
        _render(report, args.output)
        return 1
    _render(report, args.output)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
