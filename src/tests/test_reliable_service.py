import json
import base64

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.reliable_omr.definition import load_sheet_definition  # noqa: E402
from src.reliable_omr.service import create_app  # noqa: E402


def test_health_exposes_runtime_capabilities():
    response = TestClient(create_app(api_key="")).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["capabilities"]) == {
        "opencv",
        "aruco",
        "pdf",
        "calibrator_loaded",
        "api_key_required",
    }


def test_process_accepts_image_and_returns_structured_invalid_sheet():
    image = np.full((1700, 1200), 255, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success

    response = TestClient(create_app(api_key="")).post(
        "/process",
        files=[
            ("file", ("sheet.png", encoded.tobytes(), "image/png")),
            (
                "sheet_definition",
                (
                    None,
                    json.dumps(load_sheet_definition()),
                    "application/json",
                ),
            ),
            ("capture_mode", (None, "mobile", "text/plain")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "processor_version",
        "definition_id",
        "source_type",
        "sheets",
        "processing_ms",
    }
    assert body["processor_version"] == "reliable-omr/1.1.0"
    assert "processorVersion" not in body
    assert body["source_type"] == "image"
    assert len(body["sheets"]) == 1
    sheet = body["sheets"][0]
    assert sheet["status"] == "invalid"
    assert set(sheet["rectified_image"]) == {
        "jpeg_base64",
        "content_type",
        "width",
        "height",
    }
    assert sheet["rectified_image"]["content_type"] == "image/jpeg"
    assert len(
        base64.b64decode(
            sheet["rectified_image"]["jpeg_base64"], validate=True
        )
    ) <= 600_000
    assert len(sheet["questions"]) == 100
    assert set(sheet["questions"][0]["bounding_box"]) == {
        "x",
        "y",
        "width",
        "height",
    }
    assert "boundingBox" not in sheet["questions"][0]
    assert sheet["rectification"]["method"].endswith("_fallback")
    assert any(
        reason["code"] == "rectification_fallback"
        for reason in sheet["review_reasons"]
    )


def test_process_accepts_pdf_pages():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page(width=595, height=842)
    pdf_bytes = document.tobytes()
    document.close()

    response = TestClient(create_app(api_key="")).post(
        "/process",
        files={"file": ("sheet.pdf", pdf_bytes, "application/pdf")},
        data={
            "sheet_definition": json.dumps(load_sheet_definition()),
            "capture_mode": "scanner",
        },
    )

    assert response.status_code == 200
    assert response.json()["source_type"] == "pdf"
    assert len(response.json()["sheets"]) == 1


def test_process_rejects_invalid_definition_and_oversized_upload():
    client = TestClient(create_app(max_upload_bytes=10, api_key=""))
    too_large = client.post(
        "/process",
        files={"file": ("sheet.png", b"x" * 11, "image/png")},
        data={"sheet_definition": "{}", "capture_mode": "scanner"},
    )
    assert too_large.status_code == 413

    invalid = TestClient(create_app(api_key="")).post(
        "/process",
        files={"file": ("sheet.png", b"not-an-image", "image/png")},
        data={"sheet_definition": "{}", "capture_mode": "scanner"},
    )
    assert invalid.status_code == 422
    assert "Invalid sheet definition" in invalid.json()["detail"]


def test_process_api_key_is_optional_then_enforced_from_environment(monkeypatch):
    monkeypatch.delenv("OMR_API_KEY", raising=False)
    local_response = TestClient(create_app()).post(
        "/process",
        files={"file": ("sheet.png", b"not-an-image", "image/png")},
        data={"sheet_definition": "{}", "capture_mode": "scanner"},
    )
    assert local_response.status_code == 422

    monkeypatch.setenv("OMR_API_KEY", "shared-test-secret")
    client = TestClient(create_app())
    request = {
        "files": {"file": ("sheet.png", b"not-an-image", "image/png")},
        "data": {"sheet_definition": "{}", "capture_mode": "scanner"},
    }
    assert client.post("/process", **request).status_code == 401
    assert (
        client.post(
            "/process", headers={"X-Api-Key": "wrong"}, **request
        ).status_code
        == 401
    )
    accepted_auth = client.post(
        "/process",
        headers={"X-Api-Key": "shared-test-secret"},
        **request,
    )
    assert accepted_auth.status_code == 422
