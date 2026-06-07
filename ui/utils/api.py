"""Backend API client."""

import json
import logging
from typing import Any

import requests

API_URL = "http://127.0.0.1:8000"
TIMEOUT_SHORT = 5
TIMEOUT_UPLOAD = 300
TIMEOUT_STREAM = 120

logger = logging.getLogger(__name__)


def _get(path: str, timeout: int = TIMEOUT_SHORT) -> dict | list | None:
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("GET %s failed: %s", path, exc, exc_info=True)
        return None


def _post(path: str, json_data: dict | None = None, timeout: int = TIMEOUT_SHORT) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}{path}", json=json_data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _delete(path: str, timeout: int = TIMEOUT_SHORT) -> bool:
    try:
        resp = requests.delete(f"{API_URL}{path}", timeout=timeout)
        return resp.status_code in (200, 204)
    except Exception:
        return False


def is_upload_success(result: dict[str, Any]) -> bool:
    """
    Return True when the backend confirms a file was indexed.

    Accepts both "success" (upload API status) and "indexed" (legacy collision
    where doc_meta.status overwrote the upload status field).
    """
    return result.get("status") in ("success", "indexed")


def _parse_error_message(response_text: str) -> str:
    """Extract a readable message from a backend error response."""
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            detail = data.get("detail") or data.get("message")
            if detail:
                return str(detail)
    except (json.JSONDecodeError, TypeError):
        pass
    return response_text[:300] if response_text else "Unknown backend error"


def health_check() -> bool:
    data = get_health()
    if data is None:
        return False
    if data.get("status") == "ok" and not data.get("faiss_available", True):
        logger.error(
            "Backend running but FAISS unavailable — indexing will fail. "
            "Install faiss-cpu in the backend Python environment."
        )
    return data.get("status") == "ok"


def get_health() -> dict | None:
    """Return full /health payload or None."""
    data = _get("/health")
    return data if isinstance(data, dict) else None


def list_documents() -> list[dict]:
    data = _get("/documents")
    if data and "documents" in data:
        return data["documents"]
    return []


def delete_document(doc_id: str) -> bool:
    return _delete(f"/documents/{doc_id}")


def upload_document(file_name: str, file_data: bytes) -> dict[str, Any]:
    """Upload and index a PDF via POST /documents/upload."""
    logger.info(
        "Upload start: %s (%d bytes) → %s/documents/upload",
        file_name,
        len(file_data),
        API_URL,
    )
    files = [("files", (file_name, file_data, "application/pdf"))]
    try:
        resp = requests.post(
            f"{API_URL}/documents/upload",
            files=files,
            timeout=TIMEOUT_UPLOAD,
        )
    except requests.Timeout as exc:
        logger.error("Upload timed out for %s: %s", file_name, exc, exc_info=True)
        return {"status": "error", "message": "Upload timed out. The file may still be processing."}
    except requests.RequestException as exc:
        logger.error("Upload request failed for %s: %s", file_name, exc, exc_info=True)
        return {"status": "error", "message": f"Backend unreachable: {exc}"}

    logger.info(
        "Upload response %s for %s: %s",
        resp.status_code,
        file_name,
        resp.text[:500],
    )

    if resp.status_code != 200:
        message = _parse_error_message(resp.text)
        logger.error("Upload HTTP error for %s: %s", file_name, message)
        return {"status": "error", "message": message}

    try:
        payload = resp.json()
    except ValueError as exc:
        logger.error("Invalid JSON from upload endpoint for %s: %s", file_name, exc)
        return {"status": "error", "message": "Invalid response from backend"}

    results = payload.get("results", [])
    if not results:
        logger.error("Empty results array for %s", file_name)
        return {"status": "error", "message": "Empty results from backend"}

    result = results[0]
    if is_upload_success(result):
        logger.info(
            "Upload success: %s — %s chunks, %s pages (indexing complete)",
            file_name,
            result.get("chunks"),
            result.get("total_pages"),
        )
    else:
        logger.error(
            "Indexing failed for %s: %s",
            file_name,
            result.get("message", "unknown error"),
        )
    return result


def get_models() -> tuple[list[dict], str]:
    data = _get("/models")
    if not data:
        return [], ""
    return data.get("models", []), data.get("current", "")


def switch_model(model_id: str) -> bool:
    return _post("/models/switch", {"model_id": model_id}) is not None


def clear_backend_session(session_id: str) -> None:
    _delete(f"/sessions/{session_id}")


def create_backend_session(session_id: str) -> None:
    _post("/sessions", {"session_id": session_id})
