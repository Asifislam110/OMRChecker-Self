"""Versioned, dependency-light contracts for OMR modeling data."""

DATASET_SCHEMA_VERSION = "omr-capture-v1"
SPLIT_MANIFEST_VERSION = "omr-split-manifest-v1"
BUBBLE_DATASET_SCHEMA_VERSION = "omr-bubble-crop-v1"
CLASSIFIER_EXPORT_SCHEMA_VERSION = "omr-lightweight-classifier-export-v1"
CLASSIFIER_BENCHMARK_SCHEMA_VERSION = "omr-classifier-benchmark-v1"

# Keep this identical to the FastAPI/.NET capture_mode wire contract. The
# frontend describes "mobile" as guided capture, but the persisted value is
# deliberately short and stable.
CAPTURE_MODES = ("scanner", "mobile")
LABEL_STATUSES = ("verified", "adjudicated")
GROUP_FIELDS = (
    "physical_sheet_id",
    "capture_session_id",
    "device_id",
    "print_batch_id",
)

CAPTURE_REQUIRED_FIELDS = (
    "schema_version",
    "capture_id",
    "image_path",
    "physical_sheet_id",
    "capture_session_id",
    "device_id",
    "print_batch_id",
    "capture_mode",
    "is_error",
    "label_status",
)

BUBBLE_LABELS = (
    "empty",
    "filled",
    "partial",
    "crossed",
    "erased",
    "multiple",
)
BUBBLE_REQUIRED_FIELDS = (
    "schema_version",
    "crop_id",
    "capture_id",
    "physical_sheet_id",
    "capture_session_id",
    "device_id",
    "print_batch_id",
    "capture_mode",
    "question_key",
    "option_key",
    "crop_path",
    "crop_sha256",
    "label",
    "label_status",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
    "rectified_image_sha256",
    "processor_result_sha256",
    "human_annotations_sha256",
    "capture_metadata_sha256",
)

BUBBLE_EXPORT_FIELDS = (
    *BUBBLE_REQUIRED_FIELDS,
    "machine_label",
    "machine_confidence",
)

HUMAN_ANNOTATION_REQUIRED_FIELDS = (
    "capture_id",
    "question_key",
    "option_key",
    "reviewer_1_id",
    "reviewer_1_label",
    "reviewer_2_id",
    "reviewer_2_label",
    "adjudicator_id",
    "label",
    "label_status",
)

# These are data-sufficiency gates for deciding whether a benchmark is worth
# running. They are not accuracy claims or acceptance thresholds.
CLASSIFIER_REQUIRED_LABELS = ("empty", "filled")
MIN_CLASSIFIER_CROPS_PER_LABEL_MODE = 500
MIN_CLASSIFIER_SHEETS_PER_MODE = 25
MIN_CLASSIFIER_TEST_CROPS_PER_LABEL_MODE = 50

CLASSIFIER_EXPORT_REQUIRED_FIELDS = (
    "format_version",
    "model_type",
    "model_version",
    "class_names",
    "image_size",
    "feature_extractor",
    "feature_means",
    "feature_scales",
    "coefficients",
    "intercepts",
    "seed",
    "training_provenance",
    "selection",
    "metrics",
    "deployment_status",
)
