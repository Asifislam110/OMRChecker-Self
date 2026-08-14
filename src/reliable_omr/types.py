"""Typed recognition results shared by the library and HTTP service."""

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class BubbleStatus(str, Enum):
    FILLED = "filled"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


class MarkStatus(str, Enum):
    FILLED = "filled"
    EMPTY = "empty"
    MULTIPLE = "multiple"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


class SheetStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    INVALID = "invalid"


@dataclass
class ReviewReason:
    code: str
    message: str
    scope: str
    severity: str = "review"


@dataclass
class QualityIssue:
    code: str
    message: str
    severity: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class QualityReport:
    passed: bool
    metrics: Dict[str, Any]
    issues: List[QualityIssue] = field(default_factory=list)


@dataclass
class RectificationDiagnostics:
    method: str
    aruco_available: bool
    source_size_px: List[int]
    canonical_size_px: List[int]
    detected_marker_ids: List[int] = field(default_factory=list)
    missing_marker_ids: List[int] = field(default_factory=list)
    duplicate_marker_ids: List[int] = field(default_factory=list)
    marker_centers_px: Dict[str, List[float]] = field(default_factory=dict)
    transform: Optional[List[List[float]]] = None
    reprojection_error_px: Optional[float] = None
    fallback_reason: Optional[str] = None


@dataclass
class BubbleFeatures:
    inner_darkness: float
    ring_darkness: float
    background_darkness: float
    local_contrast: float
    fill_score: float
    valid_pixel_fraction: float


@dataclass
class BubbleResult:
    option: str
    center_px: List[float]
    radius_px: float
    status: BubbleStatus
    confidence: float
    features: BubbleFeatures
    review_reasons: List[ReviewReason] = field(default_factory=list)


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class QuestionResult:
    field_id: str
    question: Optional[int]
    status: MarkStatus
    answer: Optional[str]
    selected_options: List[str]
    confidence: float
    top_score: float
    second_score: float
    margin: float
    bounding_box: BoundingBox
    bubbles: List[BubbleResult]
    review_reasons: List[ReviewReason] = field(default_factory=list)


@dataclass
class ConfidenceDiagnostics:
    source: str
    risk: float
    heuristic_risk: float
    features: Dict[str, float]
    contributions: Dict[str, float]
    calibrator_version: Optional[str] = None


@dataclass
class RectifiedImage:
    jpeg_base64: str
    content_type: str
    width: int
    height: int


@dataclass
class SheetResult:
    page_number: int
    definition_id: str
    schema_version: str
    capture_mode: str
    status: SheetStatus
    confidence: float
    risk: float
    quality: QualityReport
    rectification: RectificationDiagnostics
    rectified_image: Optional[RectifiedImage]
    questions: List[QuestionResult]
    roll_number: Optional[str]
    roll_columns: List[QuestionResult]
    qr_payload: Optional[Dict[str, Any]]
    qr_raw: Optional[str]
    confidence_diagnostics: ConfidenceDiagnostics
    review_reasons: List[ReviewReason] = field(default_factory=list)


@dataclass
class DocumentResult:
    processor_version: str
    definition_id: str
    source_type: str
    sheets: List[SheetResult]
    processing_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return result_to_dict(self)


def result_to_dict(value: Any) -> Any:
    """Convert nested dataclasses/enums into JSON-safe standard Python values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            key: result_to_dict(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): result_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [result_to_dict(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
