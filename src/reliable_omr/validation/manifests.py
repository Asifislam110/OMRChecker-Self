"""Content-addressed group split manifests for leakage-safe benchmarks."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class ManifestError(ValueError):
    """Raised when a split manifest cannot safely identify benchmark splits."""


def _canonical_payload(manifest: Mapping[str, Any]) -> bytes:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return the content hash used to identify an immutable manifest."""

    return hashlib.sha256(_canonical_payload(manifest)).hexdigest()


def create_split_manifest(
    group_ids: Sequence[Any],
    group_id_field: str = "group_id",
    seed: int = 2026,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Create deterministic, disjoint group assignments."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ManifestError("calibration_fraction must be in (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ManifestError("test_fraction must be in (0, 1)")
    if calibration_fraction + test_fraction >= 1.0:
        raise ManifestError(
            "calibration_fraction + test_fraction must be below 1")
    normalized = [str(group_id).strip() for group_id in group_ids]
    if any(not group_id for group_id in normalized):
        raise ManifestError("Every group ID must be non-empty")
    unique = sorted(set(normalized))
    if len(unique) < 3:
        raise ManifestError("At least three unique groups are required")
    group_record_counts: Dict[str, int] = {}
    for group_id in normalized:
        group_record_counts[group_id] = (
            group_record_counts.get(group_id, 0) + 1
        )
    group_record_counts = dict(sorted(group_record_counts.items()))

    def order_key(group_id: str) -> Tuple[str, str]:
        encoded = "{}\0{}".format(seed, group_id).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), group_id

    ordered = sorted(unique, key=order_key)
    calibration_count = max(1, int(round(len(ordered) * calibration_fraction)))
    test_count = max(1, int(round(len(ordered) * test_fraction)))
    while calibration_count + test_count >= len(ordered):
        if calibration_count >= test_count and calibration_count > 1:
            calibration_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ManifestError(
                "Dataset is too small for three non-empty splits")

    calibration_end = calibration_count
    test_end = calibration_end + test_count
    splits = {
        "calibration": sorted(ordered[:calibration_end]),
        "test": sorted(ordered[calibration_end:test_end]),
        "train": sorted(ordered[test_end:]),
    }
    manifest: Dict[str, Any] = {
        "format_version": 1,
        "group_id_field": str(group_id_field),
        "assignment_method": "sha256(seed + NUL + group_id), fixed counts",
        "seed": int(seed),
        "group_count": len(unique),
        "record_count": len(normalized),
        "group_record_counts": group_record_counts,
        "fractions": {
            "calibration": float(calibration_fraction),
            "test": float(test_fraction),
        },
        "splits": splits,
    }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate format, content hash, uniqueness, and split disjointness."""

    if int(manifest.get("format_version", -1)) != 1:
        raise ManifestError("Unsupported split manifest format_version")
    group_id_field = str(manifest.get("group_id_field", "")).strip()
    if not group_id_field:
        raise ManifestError("Manifest group_id_field must be non-empty")
    expected_hash = str(manifest.get("manifest_sha256", ""))
    actual_hash = manifest_hash(manifest)
    if not expected_hash or expected_hash != actual_hash:
        raise ManifestError(
            "Split manifest hash mismatch; the manifest was changed "
            "or corrupted"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ManifestError("Manifest splits must be an object")
    required = {"train", "calibration", "test"}
    if set(splits) != required:
        raise ManifestError(
            "Manifest must contain exactly train/calibration/test")

    seen = set()
    for split_name in ("train", "calibration", "test"):
        values = splits[split_name]
        if not isinstance(values, list) or not values:
            raise ManifestError(
                "{} split must be a non-empty list".format(split_name))
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ManifestError("Manifest group IDs must be non-empty")
        if normalized != sorted(normalized) or len(
                normalized) != len(set(normalized)):
            raise ManifestError(
                "{} group IDs must be sorted and unique".format(split_name)
            )
        overlap = seen.intersection(normalized)
        if overlap:
            raise ManifestError(
                "Group leakage detected across splits: {}".format(
                    sorted(overlap)[:5]
                )
            )
        seen.update(normalized)
    if int(manifest.get("group_count", -1)) != len(seen):
        raise ManifestError(
            "Manifest group_count does not match split contents")
    group_record_counts = manifest.get("group_record_counts")
    if not isinstance(group_record_counts, Mapping):
        raise ManifestError("Manifest group_record_counts must be an object")
    normalized_counts = {
        str(group_id): int(count)
        for group_id, count in group_record_counts.items()
    }
    if set(normalized_counts) != seen or any(
        count < 1 for count in normalized_counts.values()
    ):
        raise ManifestError(
            "Manifest group_record_counts must cover every group"
        )
    if int(manifest.get("record_count", -1)) != sum(
        normalized_counts.values()
    ):
        raise ManifestError(
            "Manifest record_count does not match group_record_counts"
        )


def load_split_manifest(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise ManifestError("Split manifest root must be an object")
    validate_split_manifest(manifest)
    return manifest


def partition_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    require_all_manifest_groups: bool = True,
) -> Dict[str, List[Mapping[str, Any]]]:
    """Assign records from the manifest, never a mutable split column."""

    validate_split_manifest(manifest)
    group_field = str(manifest["group_id_field"])
    lookup = {}
    for split_name, group_ids in manifest["splits"].items():
        for group_id in group_ids:
            lookup[str(group_id)] = str(split_name)

    output: Dict[str, List[Mapping[str, Any]]] = {
        "train": [],
        "calibration": [],
        "test": [],
    }
    represented = set()
    for index, record in enumerate(records):
        if group_field not in record:
            raise ManifestError(
                "Record {} is missing group field '{}'".format(
                    index, group_field)
            )
        group_id = str(record[group_field]).strip()
        if group_id not in lookup:
            raise ManifestError(
                (
                    "Record group '{}' is not present in the immutable "
                    "manifest"
                ).format(group_id)
            )
        represented.add(group_id)
        output[lookup[group_id]].append(record)

    if require_all_manifest_groups:
        missing = sorted(set(lookup) - represented)
        if missing:
            raise ManifestError(
                (
                    "Benchmark omits manifest groups "
                    "(possible cherry-picking): {}"
                ).format(missing[:5])
            )
    actual_counts: Dict[str, int] = {}
    for record in records:
        group_id = str(record[group_field]).strip()
        actual_counts[group_id] = actual_counts.get(group_id, 0) + 1
    expected_counts = {
        str(group_id): int(count)
        for group_id, count in manifest["group_record_counts"].items()
    }
    if actual_counts != expected_counts:
        raise ManifestError(
            "Benchmark record counts differ from the immutable manifest "
            "(possible record-level cherry-picking)"
        )
    return output
