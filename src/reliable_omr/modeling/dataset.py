"""Audit, normalize, split, and verify physical OMR capture datasets."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.reliable_omr.calibration import CalibratorError, RISK_FEATURES
from src.reliable_omr.modeling.contracts import (
    CAPTURE_MODES,
    CAPTURE_REQUIRED_FIELDS,
    DATASET_SCHEMA_VERSION,
    GROUP_FIELDS,
    LABEL_STATUSES,
    SPLIT_MANIFEST_VERSION,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SPLIT_NAMES = ("train", "calibration", "test")


class DatasetContractError(CalibratorError):
    """Raised when modeling data violates the versioned capture contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    encoded = "\n".join(sorted(str(line) for line in lines)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(row: Mapping[str, Any], field: str, row_number: int) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise DatasetContractError(
            "Row {} field '{}' must be non-empty".format(row_number, field)
        )
    return value


def _identifier(row: Mapping[str, Any], field: str, row_number: int) -> str:
    value = _required_text(row, field, row_number)
    if not _IDENTIFIER.match(value):
        raise DatasetContractError(
            "Row {} field '{}' must be a stable identifier using only "
            "letters, numbers, '.', '_', ':', or '-'".format(row_number, field)
        )
    return value


def _binary_label(value: Any, row_number: int) -> int:
    normalized = str(value).strip()
    if normalized not in {"0", "1"}:
        raise DatasetContractError(
            "Row {} field 'is_error' must be exactly 0 or 1".format(row_number)
        )
    return int(normalized)


def _finite_float(value: Any, field: str, row_number: int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(
            "Row {} feature '{}' must be numeric".format(row_number, field)
        ) from exc
    if not math.isfinite(numeric):
        raise DatasetContractError(
            "Row {} feature '{}' must be finite".format(row_number, field)
        )
    return numeric


def _validate_risk_feature(value: float, field: str, row_number: int) -> None:
    unit_interval = {
        "rectification_fallback",
        "ambiguous_fraction",
        "multiple_fraction",
        "invalid_fraction",
        "low_margin_fraction",
        "mean_question_confidence",
        "min_question_confidence",
        "roll_uncertain_fraction",
        "qr_missing",
    }
    non_negative = {
        "quality_error_count",
        "quality_review_count",
        "reprojection_error_norm",
    }
    binary = {"rectification_fallback", "qr_missing"}
    if field in unit_interval and not 0.0 <= value <= 1.0:
        raise DatasetContractError(
            "Row {} feature '{}' must be in [0, 1]".format(
                row_number, field
            )
        )
    if field in non_negative and value < 0.0:
        raise DatasetContractError(
            "Row {} feature '{}' must be non-negative".format(
                row_number, field
            )
        )
    if field in binary and value not in {0.0, 1.0}:
        raise DatasetContractError(
            "Row {} feature '{}' must be binary".format(row_number, field)
        )


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _resolve_asset(csv_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(csv_path).parent / candidate
    return candidate.resolve()


def _read_csv(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise DatasetContractError("Capture CSV has no header")
        if len(fields) != len(set(fields)):
            raise DatasetContractError(
                "Capture CSV header has duplicate columns"
            )
        rows = list(reader)
    if not rows:
        raise DatasetContractError("Capture CSV contains no records")
    return fields, rows


def load_capture_rows(
    csv_path: Path,
    feature_names: Sequence[str] = RISK_FEATURES,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate raw or normalized capture CSV rows and verify image digests."""

    csv_path = Path(csv_path)
    fields, raw_rows = _read_csv(csv_path)
    required = set(CAPTURE_REQUIRED_FIELDS) | set(feature_names)
    missing = sorted(required - set(fields))
    if missing:
        raise DatasetContractError(
            "Capture CSV is missing columns: {}".format(missing)
        )

    capture_ids = set()
    content_hashes: Dict[str, str] = {}
    normalized: List[Dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        if None in raw:
            raise DatasetContractError(
                "Row {} has more values than header columns".format(row_index)
            )
        schema_version = _required_text(raw, "schema_version", row_index)
        if schema_version != DATASET_SCHEMA_VERSION:
            raise DatasetContractError(
                "Row {} has unsupported schema_version '{}'".format(
                    row_index, schema_version
                )
            )
        capture_id = _identifier(raw, "capture_id", row_index)
        if capture_id in capture_ids:
            raise DatasetContractError(
                "Duplicate capture_id '{}'".format(capture_id)
            )
        capture_ids.add(capture_id)

        output: Dict[str, Any] = {
            field: str(raw.get(field, "")).strip() for field in fields
        }
        output["schema_version"] = schema_version
        output["capture_id"] = capture_id
        for field in GROUP_FIELDS:
            output[field] = _identifier(raw, field, row_index)

        capture_mode = _required_text(raw, "capture_mode", row_index).lower()
        if capture_mode not in CAPTURE_MODES:
            raise DatasetContractError(
                "Row {} capture_mode must be one of {}".format(
                    row_index, list(CAPTURE_MODES)
                )
            )
        output["capture_mode"] = capture_mode
        output["is_error"] = _binary_label(raw.get("is_error"), row_index)

        label_status = _required_text(raw, "label_status", row_index).lower()
        if label_status not in LABEL_STATUSES:
            raise DatasetContractError(
                "Row {} label_status must be one of {}".format(
                    row_index, list(LABEL_STATUSES)
                )
            )
        output["label_status"] = label_status

        image_path = _required_text(raw, "image_path", row_index)
        resolved_image = _resolve_asset(csv_path, image_path)
        if not resolved_image.is_file():
            raise DatasetContractError(
                "Row {} image_path does not exist: {}".format(
                    row_index, image_path
                )
            )
        image_sha256 = sha256_file(resolved_image)
        supplied_hash = str(raw.get("capture_sha256", "")).strip().lower()
        if supplied_hash and supplied_hash != image_sha256:
            raise DatasetContractError(
                "Row {} capture_sha256 does not match image bytes".format(
                    row_index
                )
            )
        previous_capture = content_hashes.get(image_sha256)
        if previous_capture is not None:
            raise DatasetContractError(
                "Duplicate capture content for '{}' and '{}'".format(
                    previous_capture, capture_id
                )
            )
        content_hashes[image_sha256] = capture_id
        output["image_path"] = resolved_image.as_posix()
        output["capture_sha256"] = image_sha256

        for feature in feature_names:
            numeric = _finite_float(raw.get(feature), feature, row_index)
            if feature in RISK_FEATURES:
                _validate_risk_feature(numeric, feature, row_index)
            output[feature] = numeric
        normalized.append(output)

    _validate_metadata_consistency(normalized)
    normalized.sort(key=lambda row: str(row["capture_id"]))
    extra_fields = sorted(
        set(fields)
        - set(CAPTURE_REQUIRED_FIELDS)
        - set(feature_names)
        - {"capture_sha256"}
    )
    return normalized, extra_fields


def _validate_metadata_consistency(rows: Sequence[Mapping[str, Any]]) -> None:
    sheet_batches: Dict[str, str] = {}
    session_context: Dict[str, Tuple[str, str]] = {}
    for row in rows:
        sheet_id = str(row["physical_sheet_id"])
        print_batch = str(row["print_batch_id"])
        previous_batch = sheet_batches.setdefault(sheet_id, print_batch)
        if previous_batch != print_batch:
            raise DatasetContractError(
                "physical_sheet_id '{}' spans print batches '{}' and "
                "'{}'".format(sheet_id, previous_batch, print_batch)
            )

        session_id = str(row["capture_session_id"])
        context = (str(row["device_id"]), str(row["capture_mode"]))
        previous_context = session_context.setdefault(session_id, context)
        if previous_context != context:
            raise DatasetContractError(
                "capture_session_id '{}' spans multiple devices or capture "
                "modes".format(session_id)
            )


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_components(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> List[List[Mapping[str, Any]]]:
    disjoint = _DisjointSet(len(rows))
    seen: Dict[Tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for field in group_fields:
            value = str(row[field])
            key = (field, value)
            if key in seen:
                disjoint.union(index, seen[key])
            else:
                seen[key] = index
    components: Dict[int, List[Mapping[str, Any]]] = {}
    for index, row in enumerate(rows):
        components.setdefault(disjoint.find(index), []).append(row)
    return list(components.values())


def _component_id(component: Sequence[Mapping[str, Any]]) -> str:
    return sha256_lines(str(row["capture_id"]) for row in component)


def _count_values(
    rows: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _assignment_cost(
    assigned: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_split: str,
    component: Sequence[Mapping[str, Any]],
    fractions: Mapping[str, float],
    all_rows: Sequence[Mapping[str, Any]],
) -> float:
    fields = ("is_error", "capture_mode")
    cost = 0.0
    for split_name in _SPLIT_NAMES:
        split_rows = list(assigned[split_name])
        if split_name == candidate_split:
            split_rows.extend(component)
        target_total = len(all_rows) * fractions[split_name]
        cost += ((len(split_rows) - target_total) ** 2) / max(
            target_total, 1.0
        )
        for field in fields:
            global_counts = _count_values(all_rows, field)
            split_counts = _count_values(split_rows, field)
            for value, count in global_counts.items():
                target = count * fractions[split_name]
                cost += (
                    0.75
                    * ((split_counts.get(value, 0) - target) ** 2)
                    / max(target, 1.0)
                )
    return cost


def split_capture_rows(
    rows: Sequence[Mapping[str, Any]],
    seed: int = 2026,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> Dict[str, List[Mapping[str, Any]]]:
    """Create deterministic connected-group splits without metadata leakage."""

    if train_fraction <= 0 or calibration_fraction <= 0:
        raise DatasetContractError(
            "Train and calibration fractions must both be positive"
        )
    test_fraction = 1.0 - train_fraction - calibration_fraction
    if test_fraction <= 0:
        raise DatasetContractError(
            "Train + calibration fractions must be below one"
        )
    if not rows:
        raise DatasetContractError("Cannot split an empty capture dataset")
    if not group_fields:
        raise DatasetContractError(
            "At least one leakage group field is required"
        )
    missing_required_groups = sorted(set(GROUP_FIELDS) - set(group_fields))
    if missing_required_groups:
        raise DatasetContractError(
            "Leakage groups must include: {}".format(
                missing_required_groups
            )
        )
    missing_fields = sorted(
        {
            field
            for field in group_fields
            if any(field not in row for row in rows)
        }
    )
    if missing_fields:
        raise DatasetContractError(
            "Rows are missing leakage group fields: {}".format(missing_fields)
        )

    components = _group_components(rows, group_fields)
    if len(components) < 3:
        raise DatasetContractError(
            "At least three independent metadata groups are required for "
            "train/calibration/test splits; found {}".format(len(components))
        )
    fractions = {
        "train": float(train_fraction),
        "calibration": float(calibration_fraction),
        "test": float(test_fraction),
    }
    ordered_components = sorted(
        components,
        key=lambda component: (
            -len(component),
            hashlib.sha256(
                "{}:{}".format(seed, _component_id(component)).encode("utf-8")
            ).hexdigest(),
        ),
    )
    assigned: Dict[str, List[Mapping[str, Any]]] = {
        name: [] for name in _SPLIT_NAMES
    }
    for index, component in enumerate(ordered_components):
        remaining = len(ordered_components) - index
        empty_splits = [name for name in _SPLIT_NAMES if not assigned[name]]
        candidates = list(_SPLIT_NAMES)
        if remaining == len(empty_splits):
            candidates = empty_splits
        scored = []
        for split_name in candidates:
            score = _assignment_cost(
                assigned, split_name, component, fractions, rows
            )
            tie_breaker = hashlib.sha256(
                "{}:{}:{}".format(
                    seed, _component_id(component), split_name
                ).encode("utf-8")
            ).hexdigest()
            scored.append((score, tie_breaker, split_name))
        destination = min(scored)[2]
        assigned[destination].extend(component)

    for split_rows in assigned.values():
        split_rows.sort(key=lambda row: str(row["capture_id"]))
    assert_no_group_leakage(assigned, group_fields)
    return assigned


def assert_no_group_leakage(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> None:
    capture_splits: Dict[str, str] = {}
    group_splits: Dict[Tuple[str, str], str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            capture_id = str(row["capture_id"])
            previous_split = capture_splits.setdefault(capture_id, split_name)
            if previous_split != split_name:
                raise DatasetContractError(
                    "capture_id '{}' leaks across '{}' and '{}'".format(
                        capture_id, previous_split, split_name
                    )
                )
            for field in group_fields:
                key = (field, str(row[field]))
                previous_group_split = group_splits.setdefault(key, split_name)
                if previous_group_split != split_name:
                    raise DatasetContractError(
                        "{} '{}' leaks across '{}' and '{}'".format(
                            field,
                            row[field],
                            previous_group_split,
                            split_name,
                        )
                    )


def balance_report(
    rows: Sequence[Mapping[str, Any]],
    subgroup_fields: Sequence[str] = (),
) -> Dict[str, Any]:
    label_mode: Dict[str, Dict[str, int]] = {}
    for row in rows:
        mode = str(row["capture_mode"])
        label = str(row["is_error"])
        label_mode.setdefault(mode, {})
        label_mode[mode][label] = label_mode[mode].get(label, 0) + 1
    return {
        "capture_count": len(rows),
        "physical_sheet_count": len(
            {str(row["physical_sheet_id"]) for row in rows}
        ),
        "capture_session_count": len(
            {str(row["capture_session_id"]) for row in rows}
        ),
        "device_count": len({str(row["device_id"]) for row in rows}),
        "print_batch_count": len(
            {str(row["print_batch_id"]) for row in rows}
        ),
        "by_label": _count_values(rows, "is_error"),
        "by_capture_mode": _count_values(rows, "capture_mode"),
        "by_capture_mode_and_label": {
            mode: dict(sorted(counts.items()))
            for mode, counts in sorted(label_mode.items())
        },
        "subgroups": {
            field: _count_values(rows, field)
            for field in subgroup_fields
            if all(field in row for row in rows)
        },
    }


def training_readiness_report(
    splits: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    reasons = []
    for split_name in _SPLIT_NAMES:
        rows = splits.get(split_name, [])
        labels = {int(row["is_error"]) for row in rows}
        minimum = 4 if split_name in {"train", "calibration"} else 2
        if len(rows) < minimum:
            reasons.append(
                "{} split needs at least {} captures".format(
                    split_name, minimum
                )
            )
        if labels != {0, 1}:
            reasons.append(
                "{} split must contain both is_error labels".format(split_name)
            )
        modes = {str(row["capture_mode"]) for row in rows}
        missing_modes = sorted(set(CAPTURE_MODES) - modes)
        if missing_modes:
            reasons.append(
                "{} split is missing capture modes: {}".format(
                    split_name, missing_modes
                )
            )
    return {
        "ready": not reasons,
        "blocking_reasons": reasons,
        "note": (
            "Readiness only checks structural training preconditions; it does "
            "not establish representativeness or accuracy."
        ),
    }


def _normalized_fieldnames(
    feature_names: Sequence[str], extra_fields: Sequence[str]
) -> List[str]:
    return [
        *CAPTURE_REQUIRED_FIELDS,
        "capture_sha256",
        *feature_names,
        *extra_fields,
    ]


def write_normalized_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    extra_fields: Sequence[str],
) -> None:
    fieldnames = _normalized_fieldnames(feature_names, extra_fields)
    with Path(path).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for feature in feature_names:
                output[feature] = _format_float(float(row[feature]))
            output["is_error"] = str(int(row["is_error"]))
            writer.writerow(output)


def _render_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_id_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_lines(str(row["capture_id"]) for row in rows)


def prepare_capture_dataset(
    input_csv: Path,
    output_dir: Path,
    seed: int = 2026,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    feature_names: Sequence[str] = RISK_FEATURES,
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> Dict[str, Any]:
    """Audit raw data and write deterministic normalized/split artifacts."""

    input_csv = Path(input_csv).resolve()
    output_dir = Path(output_dir)
    rows, extra_fields = load_capture_rows(input_csv, feature_names)
    splits = split_capture_rows(
        rows,
        seed=seed,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        group_fields=group_fields,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / "captures.normalized.csv"
    manifest_path = output_dir / "splits.json"
    audit_path = output_dir / "audit.json"
    write_normalized_csv(
        normalized_path, rows, feature_names, extra_fields
    )

    assignment_lines = []
    split_payload = {}
    for split_name in _SPLIT_NAMES:
        split_rows = splits[split_name]
        capture_ids = [str(row["capture_id"]) for row in split_rows]
        assignment_lines.extend(
            "{}:{}".format(capture_id, split_name)
            for capture_id in capture_ids
        )
        split_payload[split_name] = {
            "capture_count": len(capture_ids),
            "capture_ids": capture_ids,
            "capture_ids_sha256": _split_id_hash(split_rows),
        }

    readiness = training_readiness_report(splits)
    manifest = {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "seed": int(seed),
        "fractions": {
            "train": float(train_fraction),
            "calibration": float(calibration_fraction),
            "test": float(1.0 - train_fraction - calibration_fraction),
        },
        "feature_names": list(feature_names),
        "group_fields": list(group_fields),
        "normalized_csv": normalized_path.name,
        "provenance": {
            "source_csv_sha256": sha256_file(input_csv),
            "normalized_csv_sha256": sha256_file(normalized_path),
            "capture_content_set_sha256": sha256_lines(
                "{}:{}".format(row["capture_id"], row["capture_sha256"])
                for row in rows
            ),
            "split_assignment_sha256": sha256_lines(assignment_lines),
        },
        "splits": split_payload,
        "training_readiness": readiness,
    }
    audit = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "source_csv": str(input_csv),
        "source_csv_sha256": sha256_file(input_csv),
        "duplicate_capture_ids": 0,
        "duplicate_capture_content": 0,
        "leakage_detected": False,
        "overall": balance_report(
            rows,
            subgroup_fields=[
                "capture_mode",
                "label_status",
                *extra_fields,
            ],
        ),
        "splits": {
            name: balance_report(
                splits[name],
                subgroup_fields=[
                    "capture_mode",
                    "label_status",
                    *extra_fields,
                ],
            )
            for name in _SPLIT_NAMES
        },
        "training_readiness": readiness,
    }
    _render_json(manifest_path, manifest)
    _render_json(audit_path, audit)
    return {
        "normalized_csv": normalized_path,
        "split_manifest": manifest_path,
        "audit_report": audit_path,
        "manifest": manifest,
        "audit": audit,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetContractError(
            "Unable to read JSON manifest '{}': {}".format(path, exc)
        ) from exc
    if not isinstance(payload, dict):
        raise DatasetContractError("Split manifest must contain a JSON object")
    return payload


def load_prepared_splits(
    normalized_csv: Path,
    split_manifest: Path,
    feature_names: Optional[Sequence[str]] = None,
    subgroup_fields: Sequence[str] = (),
    require_training_ready: bool = False,
) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Any],
]:
    """Verify prepared-data provenance and return calibrator-ready records."""

    normalized_csv = Path(normalized_csv)
    split_manifest = Path(split_manifest)
    manifest = _load_json(split_manifest)
    if manifest.get("manifest_version") != SPLIT_MANIFEST_VERSION:
        raise DatasetContractError("Unsupported split manifest version")
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetContractError("Split manifest dataset schema mismatch")
    manifest_features = manifest.get("feature_names")
    if not isinstance(manifest_features, list) or not manifest_features:
        raise DatasetContractError("Split manifest feature_names is invalid")
    selected_features = list(feature_names or manifest_features)
    if selected_features != list(manifest_features):
        raise DatasetContractError(
            "Requested features must exactly match prepared manifest features"
        )
    group_fields = manifest.get("group_fields")
    if not isinstance(group_fields, list) or not group_fields:
        raise DatasetContractError("Split manifest group_fields is invalid")
    missing_required_groups = sorted(set(GROUP_FIELDS) - set(group_fields))
    if missing_required_groups:
        raise DatasetContractError(
            "Split manifest omits required leakage groups: {}".format(
                missing_required_groups
            )
        )

    expected_csv_hash = (
        manifest.get("provenance", {}).get("normalized_csv_sha256")
    )
    if expected_csv_hash != sha256_file(normalized_csv):
        raise DatasetContractError(
            "Normalized CSV SHA-256 does not match split manifest"
        )
    rows, _ = load_capture_rows(normalized_csv, selected_features)
    missing_subgroups = sorted(
        {
            field
            for field in subgroup_fields
            if any(field not in row for row in rows)
        }
    )
    if missing_subgroups:
        raise DatasetContractError(
            "Prepared CSV is missing subgroup columns: {}".format(
                missing_subgroups
            )
        )

    rows_by_id = {str(row["capture_id"]): row for row in rows}
    assigned_ids = set()
    split_rows: Dict[str, List[Mapping[str, Any]]] = {}
    split_definitions = manifest.get("splits")
    if not isinstance(split_definitions, dict):
        raise DatasetContractError("Split manifest splits is invalid")
    for split_name in _SPLIT_NAMES:
        definition = split_definitions.get(split_name)
        if not isinstance(definition, dict):
            raise DatasetContractError(
                "Split manifest is missing '{}'".format(split_name)
            )
        capture_ids = definition.get("capture_ids")
        if not isinstance(capture_ids, list):
            raise DatasetContractError(
                "Split '{}' capture_ids must be a list".format(split_name)
            )
        if len(capture_ids) != len(set(capture_ids)):
            raise DatasetContractError(
                "Split '{}' has duplicate capture IDs".format(split_name)
            )
        unknown = sorted(set(capture_ids) - set(rows_by_id))
        if unknown:
            raise DatasetContractError(
                "Split '{}' references unknown captures: {}".format(
                    split_name, unknown[:5]
                )
            )
        overlap = assigned_ids & set(capture_ids)
        if overlap:
            raise DatasetContractError(
                "Captures appear in multiple splits: {}".format(
                    sorted(overlap)[:5]
                )
            )
        assigned_ids.update(capture_ids)
        selected_rows = [rows_by_id[capture_id] for capture_id in capture_ids]
        if definition.get("capture_count") != len(selected_rows):
            raise DatasetContractError(
                "Split '{}' capture_count does not match IDs".format(
                    split_name
                )
            )
        if (
            definition.get("capture_ids_sha256")
            != _split_id_hash(selected_rows)
        ):
            raise DatasetContractError(
                "Split '{}' ID SHA-256 does not match".format(split_name)
            )
        split_rows[split_name] = selected_rows
    missing_assignments = sorted(set(rows_by_id) - assigned_ids)
    if missing_assignments:
        raise DatasetContractError(
            "Captures missing from split manifest: {}".format(
                missing_assignments[:5]
            )
        )
    assert_no_group_leakage(split_rows, group_fields)

    content_hash = sha256_lines(
        "{}:{}".format(row["capture_id"], row["capture_sha256"])
        for row in rows
    )
    if (
        manifest.get("provenance", {}).get("capture_content_set_sha256")
        != content_hash
    ):
        raise DatasetContractError(
            "Capture-content SHA-256 does not match split manifest"
        )
    assignment_hash = sha256_lines(
        "{}:{}".format(row["capture_id"], split_name)
        for split_name, selected_rows in split_rows.items()
        for row in selected_rows
    )
    if (
        manifest.get("provenance", {}).get("split_assignment_sha256")
        != assignment_hash
    ):
        raise DatasetContractError(
            "Split-assignment SHA-256 does not match split manifest"
        )

    readiness = training_readiness_report(split_rows)
    if require_training_ready and not readiness["ready"]:
        raise DatasetContractError(
            "Prepared dataset is not training-ready: {}".format(
                "; ".join(readiness["blocking_reasons"])
            )
        )
    records: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, selected_rows in split_rows.items():
        records[split_name] = []
        for row in selected_rows:
            record = {
                "sheet_id": str(row["capture_id"]),
                "capture_id": str(row["capture_id"]),
                "physical_sheet_id": str(row["physical_sheet_id"]),
                "capture_session_id": str(row["capture_session_id"]),
                "device_id": str(row["device_id"]),
                "print_batch_id": str(row["print_batch_id"]),
                "capture_mode": str(row["capture_mode"]),
                "is_error": int(row["is_error"]),
                "source_row_count": 1,
            }
            for feature in selected_features:
                record[feature] = float(row[feature])
            for subgroup in subgroup_fields:
                record[subgroup] = str(row[subgroup])
            records[split_name].append(record)
    return records, manifest
