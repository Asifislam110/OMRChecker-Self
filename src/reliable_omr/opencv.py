"""Optional OpenCV import with a clear production error message."""

from typing import Any

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - exercised in deployments without extras
    cv2 = None  # type: Any


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for recognition. Install requirements.service.txt "
            "(opencv-contrib-python-headless provides ArUco support)."
        )
    return cv2


def has_aruco() -> bool:
    return cv2 is not None and hasattr(cv2, "aruco")
