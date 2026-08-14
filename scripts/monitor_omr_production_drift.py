"""Compare a real production window with an approved Reliable OMR baseline."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from src.reliable_omr.validation.drift import monitor_production_window
from src.reliable_omr.validation.io import read_records, write_json


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("production_records", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-window-samples", type=int, default=500)
    parser.add_argument("--minimum-field-samples", type=int, default=100)
    parser.add_argument("--minimum-outcome-samples", type=int, default=100)
    parser.add_argument("--psi-warning", type=float, default=0.1)
    parser.add_argument("--psi-alert", type=float, default=0.25)
    parser.add_argument("--tvd-warning", type=float, default=0.1)
    parser.add_argument("--tvd-alert", type=float, default=0.2)
    parser.add_argument("--target-accuracy", type=float, default=0.999)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def _render(payload, output: Optional[Path]) -> None:
    if output:
        write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        with args.baseline.open("r", encoding="utf-8") as baseline_file:
            baseline = json.load(baseline_file)
        records = read_records(args.production_records)
        report = monitor_production_window(
            baseline,
            records,
            minimum_window_samples=args.minimum_window_samples,
            minimum_field_samples=args.minimum_field_samples,
            minimum_outcome_samples=args.minimum_outcome_samples,
            psi_warning=args.psi_warning,
            psi_alert=args.psi_alert,
            tvd_warning=args.tvd_warning,
            tvd_alert=args.tvd_alert,
            target_accuracy=args.target_accuracy,
            confidence_level=args.confidence_level,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "status": "input_error",
            "error": str(exc),
        }
        _render(report, args.output)
        return 1
    _render(report, args.output)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
