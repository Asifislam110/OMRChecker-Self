"""Input and alignment quality gates used before accepting OMR output."""

from typing import Any, Dict, Mapping

import numpy as np

from src.reliable_omr.opencv import require_cv2
from src.reliable_omr.types import QualityIssue, QualityReport


CAPTURE_MODES = {"scanner", "mobile"}


def _gray(image: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Input image must be grayscale, BGR, or BGRA")


def assess_input_quality(
    image: np.ndarray,
    definition: Mapping[str, Any],
    capture_mode: str,
) -> QualityReport:
    """Measure resolution, blur, and locally saturated glare."""

    if capture_mode not in CAPTURE_MODES:
        raise ValueError(
            "capture_mode must be one of {}".format(sorted(CAPTURE_MODES))
        )

    cv2 = require_cv2()
    gray = _gray(image)
    height, width = gray.shape[:2]
    short_edge = min(height, width)
    quality = definition["quality"]

    # Variance of the Laplacian is transparent and stable enough for a gate. It
    # is not presented as a probability; capture-mode-specific thresholds live
    # in the versioned definition.
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Paper itself is frequently pure white. Count only saturated pixels that
    # are substantially brighter than their local neighbourhood.
    local_mean = cv2.GaussianBlur(
        gray, (0, 0), sigmaX=max(5, short_edge / 120.0)
    )
    paper_brightness = float(np.percentile(gray, 60))
    if paper_brightness < quality["glare_threshold"] - 5:
        # On a normally exposed mobile image, clipped pixels are distinguishable
        # from the paper baseline and their full area should count.
        glare_mask = gray >= quality["glare_threshold"]
    else:
        # Clean scanner backgrounds are often exactly 255. In that case count
        # only highlights that stand out from their local neighbourhood.
        glare_mask = (gray >= quality["glare_threshold"]) & (
            gray.astype(np.float32) - local_mean.astype(np.float32) >= 12.0
        )
    glare_fraction = float(np.mean(glare_mask))

    metrics: Dict[str, Any] = {
        "width_px": int(width),
        "height_px": int(height),
        "short_edge_px": int(short_edge),
        "blur_variance": round(blur_variance, 4),
        "glare_fraction": round(glare_fraction, 6),
        "paper_brightness_p60": round(paper_brightness, 3),
        "capture_mode": capture_mode,
    }
    issues = []

    min_edge = float(quality["min_short_edge_px"][capture_mode])
    if short_edge < min_edge:
        issues.append(
            QualityIssue(
                code="resolution_too_low",
                message="Input resolution is below the configured minimum",
                severity="error",
                value=float(short_edge),
                threshold=min_edge,
            )
        )

    min_blur = float(quality["min_blur_variance"][capture_mode])
    if blur_variance < min_blur:
        issues.append(
            QualityIssue(
                code="image_blurred",
                message="Focus score is below the configured minimum",
                severity="review",
                value=blur_variance,
                threshold=min_blur,
            )
        )

    max_glare = float(quality["max_glare_fraction"])
    if glare_fraction > max_glare:
        issues.append(
            QualityIssue(
                code="excessive_glare",
                message="Locally saturated glare exceeds the configured maximum",
                severity="review",
                value=glare_fraction,
                threshold=max_glare,
            )
        )

    return QualityReport(
        passed=not any(issue.severity == "error" for issue in issues),
        metrics=metrics,
        issues=issues,
    )


def add_marker_quality(
    report: QualityReport,
    detected_count: int,
    required_count: int,
    clipped: bool,
) -> None:
    """Append marker completeness/clipping gates to an existing report."""

    report.metrics["detected_marker_count"] = int(detected_count)
    report.metrics["required_marker_count"] = int(required_count)
    report.metrics["markers_clipped"] = bool(clipped)
    if detected_count < required_count:
        report.issues.append(
            QualityIssue(
                code="markers_incomplete",
                message="Not all required ArUco markers were detected",
                severity="error",
                value=float(detected_count),
                threshold=float(required_count),
            )
        )
    if clipped:
        report.issues.append(
            QualityIssue(
                code="marker_clipped",
                message="At least one detected marker touches the image boundary",
                severity="error",
            )
        )
    report.passed = not any(issue.severity == "error" for issue in report.issues)
