"""Accuracy validation and production monitoring for Reliable OMR."""

from src.reliable_omr.validation.drift import (
    build_drift_baseline,
    monitor_production_window,
)
from src.reliable_omr.validation.manifests import (
    create_split_manifest,
    load_split_manifest,
)
from src.reliable_omr.validation.release import build_release_report
from src.reliable_omr.validation.statistics import (
    accuracy_gate,
    clopper_pearson_lower_bound,
)

__all__ = [
    "accuracy_gate",
    "build_drift_baseline",
    "build_release_report",
    "clopper_pearson_lower_bound",
    "create_split_manifest",
    "load_split_manifest",
    "monitor_production_window",
]
