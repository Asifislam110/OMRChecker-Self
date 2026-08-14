"""FastAPI boundary for the reliable OMR library.

Run with:
    uvicorn src.reliable_omr.service:app --host 0.0.0.0 --port 8000
"""

import os
import secrets
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from src.reliable_omr.calibration import CalibratorError, SerializedRiskCalibrator
from src.reliable_omr.definition import SheetDefinitionError
from src.reliable_omr.opencv import cv2, has_aruco
from src.reliable_omr.recognizer import (
    PROCESSOR_VERSION,
    RecognitionInputError,
    ReliableOMRRecognizer,
)


DEFAULT_MAX_UPLOAD_BYTES = 30 * 1024 * 1024


def _load_configured_calibrator(
    source: Optional[Union[str, Path, SerializedRiskCalibrator]]
) -> Optional[SerializedRiskCalibrator]:
    if isinstance(source, SerializedRiskCalibrator):
        return source
    configured = source or os.getenv("OMR_CALIBRATOR_PATH")
    if not configured:
        return None
    return SerializedRiskCalibrator.load(configured)


def create_app(
    calibrator: Optional[
        Union[str, Path, SerializedRiskCalibrator]
    ] = None,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    include_rectified_image: bool = True,
    api_key: Optional[str] = None,
) -> FastAPI:
    configured_api_key = (
        api_key if api_key is not None else os.getenv("OMR_API_KEY")
    )
    if configured_api_key is not None and not configured_api_key.strip():
        configured_api_key = None
    app = FastAPI(
        title="Reliable OMR Service",
        version=PROCESSOR_VERSION,
        description=(
            "Classical OpenCV OMR with versioned geometry, explicit quality "
            "gates, and review-first confidence routing."
        ),
    )
    try:
        configured_calibrator = _load_configured_calibrator(calibrator)
        calibrator_error = None
    except (OSError, ValueError, CalibratorError) as exc:
        configured_calibrator = None
        calibrator_error = str(exc)

    @app.get("/health")
    async def health() -> Any:
        capabilities = {
            "opencv": cv2 is not None,
            "aruco": has_aruco(),
            "pdf": _module_available("pymupdf") or _module_available("fitz"),
            "calibrator_loaded": configured_calibrator is not None,
            "api_key_required": configured_api_key is not None,
        }
        errors = []
        if cv2 is None:
            errors.append("opencv_unavailable")
        elif not has_aruco():
            errors.append("aruco_unavailable")
        if calibrator_error:
            errors.append("calibrator_invalid")
        return {
            "status": "ok" if not errors else "degraded",
            "service_version": PROCESSOR_VERSION,
            "capabilities": capabilities,
            "errors": errors,
            "calibrator_error": calibrator_error,
        }

    @app.post("/process")
    async def process(
        file: UploadFile = File(...),
        sheet_definition: str = Form(...),
        capture_mode: str = Form("scanner"),
        x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    ) -> Any:
        if configured_api_key is not None and not secrets.compare_digest(
            x_api_key or "", configured_api_key
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        content_type = (file.content_type or "").lower()
        if content_type and not (
            content_type.startswith("image/")
            or content_type
            in {"application/pdf", "application/octet-stream"}
        ):
            raise HTTPException(
                status_code=415,
                detail="file must be an image or PDF",
            )
        data = await file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="file exceeds the {} byte limit".format(
                    max_upload_bytes
                ),
            )
        try:
            recognizer = ReliableOMRRecognizer(
                sheet_definition=sheet_definition,
                calibrator=configured_calibrator,
                include_rectified_image=include_rectified_image,
            )
            return recognizer.recognize(
                data, capture_mode=capture_mode
            ).to_dict()
        except SheetDefinitionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RecognitionInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


app = create_app()
