"""Bounded, metadata-free JPEG previews of canonical rectified sheets."""

import base64
from typing import Tuple

import numpy as np

from src.reliable_omr.opencv import require_cv2
from src.reliable_omr.types import RectifiedImage


DEFAULT_PREVIEW_MAX_WIDTH = 1000
DEFAULT_PREVIEW_MAX_HEIGHT = 1415
DEFAULT_PREVIEW_MAX_JPEG_BYTES = 600_000


def encode_rectified_preview(
    image: np.ndarray,
    max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
    max_height: int = DEFAULT_PREVIEW_MAX_HEIGHT,
    max_jpeg_bytes: int = DEFAULT_PREVIEW_MAX_JPEG_BYTES,
) -> RectifiedImage:
    """Downscale and re-encode a preview under a strict binary byte limit."""

    if min(max_width, max_height, max_jpeg_bytes) <= 0:
        raise ValueError("Preview dimensions and byte limit must be positive")
    cv2 = require_cv2()
    source_height, source_width = image.shape[:2]
    scale = min(
        1.0,
        max_width / float(source_width),
        max_height / float(source_height),
    )
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    preview = (
        image
        if (width, height) == (source_width, source_height)
        else cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    )

    encoded = None
    # Quality reduction handles ordinary high-detail pages. Dimension reduction
    # provides a deterministic hard-bound fallback for adversarial noise.
    for _ in range(8):
        encoded = _encode_with_qualities(
            preview, (76, 66, 56, 46), max_jpeg_bytes
        )
        if encoded is not None:
            break
        width = max(64, int(round(width * 0.8)))
        height = max(64, int(round(height * 0.8)))
        preview = cv2.resize(
            preview, (width, height), interpolation=cv2.INTER_AREA
        )

    if encoded is None:
        success, buffer = cv2.imencode(
            ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 35]
        )
        if not success or len(buffer) > max_jpeg_bytes:
            raise RuntimeError("Could not create a bounded rectified preview")
        encoded = buffer.tobytes()

    return RectifiedImage(
        jpeg_base64=base64.b64encode(encoded).decode("ascii"),
        content_type="image/jpeg",
        width=int(preview.shape[1]),
        height=int(preview.shape[0]),
    )


def _encode_with_qualities(
    image: np.ndarray,
    qualities: Tuple[int, ...],
    max_jpeg_bytes: int,
):
    cv2 = require_cv2()
    for quality in qualities:
        success, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            continue
        if len(buffer) <= max_jpeg_bytes:
            return buffer.tobytes()
    return None
