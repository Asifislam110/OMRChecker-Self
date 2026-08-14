import base64

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.reliable_omr.preview import encode_rectified_preview  # noqa: E402


def test_preview_is_deterministic_jpeg_and_respects_hard_bounds():
    randomizer = np.random.default_rng(2026)
    image = randomizer.integers(
        0, 256, size=(900, 700), dtype=np.uint8
    )

    first = encode_rectified_preview(
        image, max_width=500, max_height=700, max_jpeg_bytes=50_000
    )
    second = encode_rectified_preview(
        image, max_width=500, max_height=700, max_jpeg_bytes=50_000
    )
    binary = base64.b64decode(first.jpeg_base64, validate=True)

    assert first == second
    assert first.content_type == "image/jpeg"
    assert first.width <= 500
    assert first.height <= 700
    assert len(binary) <= 50_000
    assert binary.startswith(b"\xff\xd8")
