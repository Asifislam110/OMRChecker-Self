"""Contracts and tooling for representative, leakage-safe OMR model data."""

from src.reliable_omr.modeling.contracts import (
    BUBBLE_DATASET_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    SPLIT_MANIFEST_VERSION,
)
from src.reliable_omr.modeling.dataset import (
    DatasetContractError,
    load_prepared_splits,
    prepare_capture_dataset,
)
from src.reliable_omr.modeling.labeling import (
    export_bubble_crops,
    export_capture_row,
)
from src.reliable_omr.modeling.readiness import classifier_readiness_report

__all__ = [
    "BUBBLE_DATASET_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "DatasetContractError",
    "SPLIT_MANIFEST_VERSION",
    "classifier_readiness_report",
    "export_bubble_crops",
    "export_capture_row",
    "load_prepared_splits",
    "prepare_capture_dataset",
]
