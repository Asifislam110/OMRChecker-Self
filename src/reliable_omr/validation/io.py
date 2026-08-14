"""Small, explicit readers and writers for validation command-line tools."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


class ValidationInputError(ValueError):
    """Raised when benchmark or monitoring input is not machine-validatable."""


def read_records(path: Path) -> List[Dict[str, Any]]:
    """Read CSV, JSON array/object, or newline-delimited JSON records."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            records = [dict(row) for row in csv.DictReader(input_file)]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, Mapping):
            payload = payload.get("records")
        if not isinstance(payload, list):
            raise ValidationInputError(
                "JSON input must be an array or an object containing 'records'"
            )
        records = [dict(record)
                   for record in payload if isinstance(record, Mapping)]
        if len(records) != len(payload):
            raise ValidationInputError("Every JSON record must be an object")
    elif suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationInputError(
                        "Invalid JSON on line {}".format(line_number)
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValidationInputError(
                        "JSONL line {} must contain an object".format(
                            line_number)
                    )
                records.append(dict(record))
    else:
        raise ValidationInputError(
            "Input must use .csv, .json, .jsonl, or .ndjson"
        )
    if not records:
        raise ValidationInputError("Input contains no records")
    return records


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON without silently creating directories."""

    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    Path(path).write_text(rendered, encoding="utf-8")


def parse_binary(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValidationInputError(
        "{} must contain only true/false or 1/0".format(field_name)
    )


def finite_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationInputError(
            "{} must contain numeric values".format(field_name)
        ) from exc
    if not math.isfinite(parsed):
        raise ValidationInputError(
            "{} values must be finite".format(field_name))
    return parsed


def unique_strings(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values))
