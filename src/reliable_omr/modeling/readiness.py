"""Data and dependency gates for the shadow lightweight bubble benchmark."""

import csv
import importlib.util
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.reliable_omr.modeling.contracts import (
    BUBBLE_DATASET_SCHEMA_VERSION,
    BUBBLE_LABELS,
    BUBBLE_REQUIRED_FIELDS,
    CLASSIFIER_BENCHMARK_SCHEMA_VERSION,
    CLASSIFIER_EXPORT_REQUIRED_FIELDS,
    CLASSIFIER_EXPORT_SCHEMA_VERSION,
    CLASSIFIER_REQUIRED_LABELS,
    GROUP_FIELDS,
    LABEL_STATUSES,
    MIN_CLASSIFIER_CROPS_PER_LABEL_MODE,
    MIN_CLASSIFIER_SHEETS_PER_MODE,
    MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE,
)
from src.reliable_omr.modeling.dataset import (
    DatasetContractError,
    load_capture_rows,
    load_prepared_splits,
    sha256_file,
)


def _dependency_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _dependency_declared(package_prefix: str) -> bool:
    project_root = Path(__file__).resolve().parents[3]
    normalized_prefix = package_prefix.lower().replace("_", "-")
    for requirements_path in project_root.glob("requirements*.txt"):
        for line in requirements_path.read_text(encoding="utf-8").splitlines():
            normalized = line.strip().lower().replace("_", "-")
            if normalized.startswith(normalized_prefix):
                return True
    return False


def _classifier_contract() -> Dict[str, Any]:
    return {
        "candidate": "hog_multinomial_logistic",
        "input": {
            "schema_version": BUBBLE_DATASET_SCHEMA_VERSION,
            "required_csv_fields": list(BUBBLE_REQUIRED_FIELDS),
            "allowed_labels": list(BUBBLE_LABELS),
            "split_source": (
                "Each crop inherits its parent capture's split; crop-level "
                "random splitting is forbidden."
            ),
        },
        "export": {
            "format_version": CLASSIFIER_EXPORT_SCHEMA_VERSION,
            "required_fields": list(CLASSIFIER_EXPORT_REQUIRED_FIELDS),
            "model_type": "hog_multinomial_logistic",
            "image_size": [32, 32],
            "feature_extractor": {
                "name": "opencv_hog",
                "win_size": [32, 32],
                "block_size": [16, 16],
                "block_stride": [8, 8],
                "cell_size": [8, 8],
                "bins": 9,
            },
            "serialization": (
                "JSON numeric parameters only; pickle/joblib exports are not "
                "accepted."
            ),
        },
        "benchmark_output": {
            "schema_version": CLASSIFIER_BENCHMARK_SCHEMA_VERSION,
            "required_provenance": [
                "normalized_capture_csv_sha256",
                "split_manifest_sha256",
                "bubble_manifest_sha256",
            ],
            "required_metrics": [
                "per_class_precision_recall",
                "balanced_accuracy",
                "confusion_matrix",
                "scanner_subgroup",
                "mobile_subgroup",
                "physical_sheet_count",
            ],
            "note": (
                "Metrics must come only from the untouched test split and are "
                "not populated by the readiness command."
            ),
        },
    }


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_verified_bubble_rows(
    bubble_csv: Path,
    captures: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> List[Dict[str, Any]]:
    with Path(bubble_csv).open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise DatasetContractError("Bubble CSV has no header")
        if len(fields) != len(set(fields)):
            raise DatasetContractError(
                "Bubble CSV header has duplicate columns"
            )
        missing = sorted(set(BUBBLE_REQUIRED_FIELDS) - set(fields))
        if missing:
            raise DatasetContractError(
                "Bubble CSV is missing columns: {}".format(missing)
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise DatasetContractError("Bubble CSV contains no records")

    crop_ids = set()
    content_hashes: Dict[str, str] = {}
    capture_provenance: Dict[str, Tuple[str, ...]] = {}
    output = []
    for row_number, raw in enumerate(raw_rows, start=2):
        if None in raw:
            raise DatasetContractError(
                "Bubble row {} has more values than header columns".format(
                    row_number
                )
            )
        schema_version = str(raw.get("schema_version", "")).strip()
        if schema_version != BUBBLE_DATASET_SCHEMA_VERSION:
            raise DatasetContractError(
                "Bubble row {} has unsupported schema_version".format(
                    row_number
                )
            )
        crop_id = str(raw.get("crop_id", "")).strip()
        if not _IDENTIFIER.fullmatch(crop_id):
            raise DatasetContractError(
                "Bubble row {} crop_id must be a stable identifier".format(
                    row_number
                )
            )
        if crop_id in crop_ids:
            raise DatasetContractError(
                "Duplicate bubble crop_id '{}'".format(crop_id)
            )
        crop_ids.add(crop_id)

        capture_id = str(raw.get("capture_id", "")).strip()
        if capture_id not in captures:
            raise DatasetContractError(
                "Bubble crop '{}' references unknown capture_id '{}'".format(
                    crop_id, capture_id
                )
            )
        parent = captures[capture_id]
        inherited = {}
        for field in GROUP_FIELDS:
            value = str(raw.get(field, "")).strip()
            if value != str(parent[field]):
                raise DatasetContractError(
                    "Bubble crop '{}' {} disagrees with parent capture".format(
                        crop_id, field
                    )
                )
            inherited[field] = value
        capture_mode = str(raw.get("capture_mode", "")).strip().lower()
        if capture_mode != str(parent["capture_mode"]):
            raise DatasetContractError(
                "Bubble crop '{}' capture_mode disagrees with parent "
                "capture".format(crop_id)
            )

        for field in ("question_key", "option_key"):
            value = str(raw.get(field, "")).strip()
            if not _IDENTIFIER.fullmatch(value):
                raise DatasetContractError(
                    "Bubble crop '{}' {} is not a valid identifier".format(
                        crop_id, field
                    )
                )
        label = str(raw.get("label", "")).strip().lower()
        if label not in BUBBLE_LABELS:
            raise DatasetContractError(
                "Bubble crop '{}' label must be one of {}".format(
                    crop_id, list(BUBBLE_LABELS)
                )
            )
        label_status = str(raw.get("label_status", "")).strip().lower()
        if label_status not in LABEL_STATUSES:
            raise DatasetContractError(
                "Bubble crop '{}' label_status must be one of {}".format(
                    crop_id, list(LABEL_STATUSES)
                )
            )
        crop_path_value = str(raw.get("crop_path", "")).strip()
        if not crop_path_value:
            raise DatasetContractError(
                "Bubble crop '{}' crop_path must be non-empty".format(crop_id)
            )
        crop_path = Path(crop_path_value)
        if not crop_path.is_absolute():
            crop_path = Path(bubble_csv).parent / crop_path
        if not crop_path.is_file():
            raise DatasetContractError(
                "Bubble crop '{}' file does not exist: {}".format(
                    crop_id, crop_path_value
                )
            )
        crop_sha256 = sha256_file(crop_path)
        supplied_hash = str(raw.get("crop_sha256", "")).strip().lower()
        if supplied_hash != crop_sha256:
            raise DatasetContractError(
                "Bubble crop '{}' SHA-256 does not match file bytes".format(
                    crop_id
                )
            )
        previous_crop = content_hashes.get(crop_sha256)
        if previous_crop is not None:
            raise DatasetContractError(
                "Duplicate bubble content for '{}' and '{}'".format(
                    previous_crop, crop_id
                )
            )
        content_hashes[crop_sha256] = crop_id
        provenance_fields = (
            "rectified_image_sha256",
            "processor_result_sha256",
            "human_annotations_sha256",
            "capture_metadata_sha256",
        )
        provenance_values = tuple(
            str(raw.get(field, "")).strip().lower()
            for field in provenance_fields
        )
        for field, value in zip(provenance_fields, provenance_values):
            if not _SHA256.fullmatch(value):
                raise DatasetContractError(
                    "Bubble crop '{}' {} must be a SHA-256 digest".format(
                        crop_id, field
                    )
                )
        previous_provenance = capture_provenance.setdefault(
            capture_id, provenance_values
        )
        if previous_provenance != provenance_values:
            raise DatasetContractError(
                "Bubble crops for capture '{}' have inconsistent source "
                "provenance".format(capture_id)
            )
        bbox = {}
        for field in ("bbox_x", "bbox_y", "bbox_width", "bbox_height"):
            try:
                bbox[field] = int(str(raw.get(field, "")).strip())
            except ValueError as exc:
                raise DatasetContractError(
                    "Bubble crop '{}' {} must be an integer".format(
                        crop_id, field
                    )
                ) from exc
        if (
            bbox["bbox_x"] < 0
            or bbox["bbox_y"] < 0
            or bbox["bbox_width"] <= 0
            or bbox["bbox_height"] <= 0
        ):
            raise DatasetContractError(
                "Bubble crop '{}' has an invalid source bounding box".format(
                    crop_id
                )
            )
        machine_label = str(raw.get("machine_label", "")).strip().lower()
        machine_confidence = None
        if machine_label:
            if machine_label not in {
                "empty",
                "filled",
                "ambiguous",
                "invalid",
            }:
                raise DatasetContractError(
                    "Bubble crop '{}' has invalid machine_label".format(
                        crop_id
                    )
                )
            try:
                machine_confidence = float(raw.get("machine_confidence", ""))
            except (TypeError, ValueError) as exc:
                raise DatasetContractError(
                    "Bubble crop '{}' machine_confidence must be "
                    "numeric".format(
                        crop_id
                    )
                ) from exc
            if (
                not math.isfinite(machine_confidence)
                or not 0.0 <= machine_confidence <= 1.0
            ):
                raise DatasetContractError(
                    "Bubble crop '{}' machine_confidence must be in "
                    "[0, 1]".format(
                        crop_id
                    )
                )
        output.append(
            {
                "crop_id": crop_id,
                "capture_id": capture_id,
                **inherited,
                "capture_mode": capture_mode,
                "question_key": str(raw["question_key"]).strip(),
                "option_key": str(raw["option_key"]).strip(),
                "label": label,
                "label_status": label_status,
                "crop_path": crop_path_value.replace("\\", "/"),
                "resolved_crop_path": crop_path.resolve(),
                "crop_sha256": crop_sha256,
                "split": assignment[capture_id],
                "machine_label": machine_label or None,
                "machine_confidence": machine_confidence,
            }
        )
    output.sort(key=lambda row: str(row["crop_id"]))
    return output


def _count_pairs(
    rows: Sequence[Mapping[str, Any]],
    left_field: str,
    right_field: str,
) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        left = str(row[left_field])
        right = str(row[right_field])
        counts.setdefault(left, {})
        counts[left][right] = counts[left].get(right, 0) + 1
    return {
        left: dict(sorted(values.items()))
        for left, values in sorted(counts.items())
    }


def _data_gates(
    rows: Sequence[Mapping[str, Any]]
) -> Tuple[List[str], Dict[str, Any]]:
    reasons = []
    label_mode = _count_pairs(rows, "capture_mode", "label")
    sheet_counts: Dict[str, int] = {}
    test_rows = [row for row in rows if row["split"] == "test"]
    test_label_mode = _count_pairs(test_rows, "capture_mode", "label")
    for mode in ("scanner", "mobile"):
        sheet_counts[mode] = len(
            {
                str(row["physical_sheet_id"])
                for row in rows
                if row["capture_mode"] == mode
            }
        )
        if sheet_counts[mode] < MIN_CLASSIFIER_SHEETS_PER_MODE:
            reasons.append(
                "{} needs at least {} independent physical sheets; found "
                "{}".format(
                    mode,
                    MIN_CLASSIFIER_SHEETS_PER_MODE,
                    sheet_counts[mode],
                )
            )
        for label in CLASSIFIER_REQUIRED_LABELS:
            count = label_mode.get(mode, {}).get(label, 0)
            if count < MIN_CLASSIFIER_CROPS_PER_LABEL_MODE:
                reasons.append(
                    "{} / {} needs at least {} crops; found {}".format(
                        mode,
                        label,
                        MIN_CLASSIFIER_CROPS_PER_LABEL_MODE,
                        count,
                    )
                )
            test_count = test_label_mode.get(mode, {}).get(label, 0)
            if test_count < MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE:
                reasons.append(
                    "test split {} / {} needs at least {} crops; found "
                    "{}".format(
                        mode,
                        label,
                        MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE,
                        test_count,
                    )
                )
    return reasons, {
        "crop_count": len(rows),
        "physical_sheets_by_mode": sheet_counts,
        "crops_by_mode_and_label": label_mode,
        "test_crops_by_mode_and_label": test_label_mode,
        "gates": {
            "minimum_crops_per_required_label_mode": (
                MIN_CLASSIFIER_CROPS_PER_LABEL_MODE
            ),
            "minimum_physical_sheets_per_mode": (
                MIN_CLASSIFIER_SHEETS_PER_MODE
            ),
            "minimum_test_crops_per_required_label_mode": (
                MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE
            ),
        },
    }


def classifier_readiness_report(
    normalized_csv: Optional[Path] = None,
    split_manifest: Optional[Path] = None,
    bubble_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a gate report without training or inventing benchmark metrics."""

    dependencies = {
        "opencv_hog": {
            "runtime_available": _dependency_available("cv2"),
            "declared_project_dependency": _dependency_declared("opencv"),
        },
        "sklearn_logistic_regression": {
            "runtime_available": _dependency_available("sklearn"),
            "declared_project_dependency": _dependency_declared(
                "scikit-learn"
            ),
        },
    }
    reasons = []
    data_summary: Dict[str, Any] = {
        "crop_count": 0,
        "physical_sheets_by_mode": {},
        "crops_by_mode_and_label": {},
        "test_crops_by_mode_and_label": {},
    }
    provenance: Dict[str, str] = {}
    if bubble_csv is None:
        reasons.append("no verified bubble-crop manifest was supplied")
    elif normalized_csv is None or split_manifest is None:
        reasons.append(
            "bubble readiness requires normalized captures and a split "
            "manifest"
        )
    else:
        prepared, _ = load_prepared_splits(
            normalized_csv,
            split_manifest,
            require_training_ready=False,
        )
        capture_rows, _ = load_capture_rows(normalized_csv)
        captures = {
            str(row["capture_id"]): row for row in capture_rows
        }
        assignment = {
            str(record["capture_id"]): split_name
            for split_name, records in prepared.items()
            for record in records
        }
        bubble_rows = load_verified_bubble_rows(
            Path(bubble_csv), captures, assignment
        )
        gate_reasons, data_summary = _data_gates(bubble_rows)
        reasons.extend(gate_reasons)
        provenance = {
            "normalized_capture_csv_sha256": sha256_file(normalized_csv),
            "split_manifest_sha256": sha256_file(split_manifest),
            "bubble_manifest_sha256": sha256_file(bubble_csv),
        }

    missing_dependencies = [
        name
        for name, status in dependencies.items()
        if not status["runtime_available"]
        or not status["declared_project_dependency"]
    ]
    if missing_dependencies:
        reasons.append(
            "missing optional benchmark dependencies: {}".format(
                missing_dependencies
            )
        )
    ready = not reasons
    return {
        "status": "ready_for_benchmark" if ready else "blocked",
        "benchmark_ready": ready,
        "blocking_reasons": reasons,
        "dependencies": dependencies,
        "data_summary": data_summary,
        "provenance": provenance,
        "contract": _classifier_contract(),
        "measured_metrics": None,
        "note": (
            "No classifier is trained and no accuracy is claimed by this "
            "readiness report."
        ),
    }
