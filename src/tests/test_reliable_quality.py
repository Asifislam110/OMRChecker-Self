import numpy as np
import pytest

pytest.importorskip("cv2")

from src.reliable_omr.definition import load_sheet_definition  # noqa: E402
from src.reliable_omr.quality import (  # noqa: E402
    add_marker_quality,
    assess_input_quality,
)


def test_resolution_and_blur_gates_report_measured_values():
    image = np.full((800, 600), 255, dtype=np.uint8)

    report = assess_input_quality(
        image, load_sheet_definition(), capture_mode="mobile"
    )

    assert report.passed is False
    assert report.metrics["short_edge_px"] == 600
    assert report.metrics["blur_variance"] == 0
    assert {issue.code for issue in report.issues} >= {
        "resolution_too_low",
        "image_blurred",
    }


def test_mobile_glare_and_marker_clipping_are_explicit_gates():
    image = np.full((1200, 1200), 180, dtype=np.uint8)
    image[:, :240] = 255

    report = assess_input_quality(
        image, load_sheet_definition(), capture_mode="mobile"
    )
    add_marker_quality(report, detected_count=4, required_count=4, clipped=True)

    assert report.metrics["glare_fraction"] == pytest.approx(0.2)
    assert {issue.code for issue in report.issues} >= {
        "excessive_glare",
        "marker_clipped",
    }
    assert report.passed is False
