"""ArUco-based page rectification into the definition's canonical pixels."""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np

from src.reliable_omr.definition import point_mm_to_px
from src.reliable_omr.opencv import has_aruco, require_cv2
from src.reliable_omr.types import RectificationDiagnostics


@dataclass
class RectificationResult:
    image: np.ndarray
    diagnostics: RectificationDiagnostics
    markers_clipped: bool


def _canonical_marker_corners(
    marker: Mapping[str, Any], definition: Mapping[str, Any]
) -> np.ndarray:
    x, y = marker["top_left_mm"]
    size = marker["size_mm"]
    points_mm = [
        [x, y],
        [x + size, y],
        [x + size, y + size],
        [x, y + size],
    ]
    return np.asarray(
        [point_mm_to_px(point, definition) for point in points_mm],
        dtype=np.float32,
    )


def _resize_fallback(
    image: np.ndarray,
    definition: Mapping[str, Any],
    aruco_available: bool,
    reason: str,
    detected_ids: List[int] = None,
    missing_ids: List[int] = None,
    duplicate_ids: List[int] = None,
    marker_centers: Dict[str, List[float]] = None,
) -> RectificationResult:
    cv2 = require_cv2()
    page = definition["page"]
    width = int(page["canonical_width_px"])
    height = int(page["canonical_height_px"])
    source_height, source_width = image.shape[:2]
    source_ratio = source_width / float(source_height)
    target_ratio = width / float(height)
    ratio_error = abs(source_ratio - target_ratio) / target_ratio
    method = (
        "aspect_resize_fallback"
        if ratio_error <= 0.04
        else "unrectified_resize_fallback"
    )
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    diagnostics = RectificationDiagnostics(
        method=method,
        aruco_available=aruco_available,
        source_size_px=[int(source_width), int(source_height)],
        canonical_size_px=[width, height],
        detected_marker_ids=detected_ids or [],
        missing_marker_ids=missing_ids or [],
        duplicate_marker_ids=duplicate_ids or [],
        marker_centers_px=marker_centers or {},
        fallback_reason=reason,
    )
    return RectificationResult(resized, diagnostics, False)


def rectify_to_canonical(
    image: np.ndarray, definition: Mapping[str, Any]
) -> RectificationResult:
    """Detect four configured markers and fit a 16-point page homography.

    If the OpenCV build lacks ``cv2.aruco`` or markers are incomplete, a
    deterministic resize is returned so callers receive structured diagnostics
    instead of a crash. The recognizer always routes that fallback to review or
    invalid rather than silently treating it as equivalent to rectification.
    """

    cv2 = require_cv2()
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if not has_aruco():
        expected = [marker["id"] for marker in definition["aruco"]["markers"]]
        return _resize_fallback(
            image,
            definition,
            aruco_available=False,
            reason="cv2.aruco is unavailable; install opencv-contrib-python",
            missing_ids=expected,
        )

    aruco = cv2.aruco
    dictionary_name = definition["aruco"]["dictionary"]
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        return _resize_fallback(
            image,
            definition,
            aruco_available=True,
            reason="OpenCV does not provide ArUco dictionary {}".format(
                dictionary_name
            ),
            missing_ids=[
                marker["id"] for marker in definition["aruco"]["markers"]
            ],
        )

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:  # OpenCV 4.6 and older
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        corners, ids, _ = aruco.ArucoDetector(
            dictionary, parameters
        ).detectMarkers(gray)
    else:  # OpenCV 4.6 and older
        corners, ids, _ = aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )

    detected_ids = [] if ids is None else [int(value) for value in ids.flatten()]
    counts = Counter(detected_ids)
    duplicate_ids = sorted(
        marker_id for marker_id, count in counts.items() if count > 1
    )
    expected_by_id = {
        int(marker["id"]): marker for marker in definition["aruco"]["markers"]
    }
    expected_ids = sorted(expected_by_id)
    missing_ids = sorted(set(expected_ids) - set(detected_ids))

    corner_by_id: Dict[int, np.ndarray] = {}
    for index, marker_id in enumerate(detected_ids):
        if marker_id in expected_by_id and marker_id not in corner_by_id:
            corner_by_id[marker_id] = np.asarray(
                corners[index], dtype=np.float32
            ).reshape(4, 2)
    marker_centers = {
        str(marker_id): [
            round(float(value), 3) for value in points.mean(axis=0)
        ]
        for marker_id, points in corner_by_id.items()
    }

    if missing_ids or duplicate_ids:
        reason_parts = []
        if missing_ids:
            reason_parts.append("missing markers {}".format(missing_ids))
        if duplicate_ids:
            reason_parts.append("duplicate markers {}".format(duplicate_ids))
        return _resize_fallback(
            image,
            definition,
            aruco_available=True,
            reason="; ".join(reason_parts),
            detected_ids=sorted(set(detected_ids)),
            missing_ids=missing_ids,
            duplicate_ids=duplicate_ids,
            marker_centers=marker_centers,
        )

    source_points = []
    destination_points = []
    for marker_id in expected_ids:
        source_points.extend(corner_by_id[marker_id])
        destination_points.extend(
            _canonical_marker_corners(expected_by_id[marker_id], definition)
        )
    source = np.asarray(source_points, dtype=np.float32)
    destination = np.asarray(destination_points, dtype=np.float32)
    transform, _ = cv2.findHomography(source, destination, method=0)
    if transform is None or not np.isfinite(transform).all():
        return _resize_fallback(
            image,
            definition,
            aruco_available=True,
            reason="OpenCV could not fit a finite marker homography",
            detected_ids=expected_ids,
            marker_centers=marker_centers,
        )

    projected = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected - destination) ** 2, axis=1)))
    )
    page = definition["page"]
    width = int(page["canonical_width_px"])
    height = int(page["canonical_height_px"])
    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    source_height, source_width = image.shape[:2]
    edge_margin = max(2.0, min(source_height, source_width) * 0.001)
    markers_clipped = bool(
        np.any(source[:, 0] <= edge_margin)
        or np.any(source[:, 1] <= edge_margin)
        or np.any(source[:, 0] >= source_width - 1 - edge_margin)
        or np.any(source[:, 1] >= source_height - 1 - edge_margin)
    )
    diagnostics = RectificationDiagnostics(
        method="aruco_homography",
        aruco_available=True,
        source_size_px=[int(source_width), int(source_height)],
        canonical_size_px=[width, height],
        detected_marker_ids=expected_ids,
        marker_centers_px=marker_centers,
        transform=[
            [round(float(value), 10) for value in row] for row in transform
        ],
        reprojection_error_px=round(reprojection_error, 4),
    )
    return RectificationResult(warped, diagnostics, markers_clipped)
