"""SSE stream consumer for the Streamlit frontend."""

from __future__ import annotations

import json
import logging
from typing import Callable

import requests

from shared.sse_codec import parse_sse_data_line
from shared.stream_assembler import StreamAssembler
from ui.utils.api import API_URL, TIMEOUT_STREAM

logger = logging.getLogger(__name__)

_DIAGNOSTIC_CHUNK_LIMIT = 50


def consume_chat_stream(
    response: requests.Response,
    on_token: Callable[[str, str], None] | None = None,
) -> tuple[str, dict]:
    """
    Read an SSE chat response and assemble tokens.

    Returns:
        (full_response_text, sources_dict)
    """
    assembler = StreamAssembler()
    sources: dict = {}
    diagnostic_count = 0

    # Buffer for multi-line SSE events (payload lines joined with \\n)
    event_lines: list[str] = []

    def flush_event() -> None:
        nonlocal diagnostic_count
        if not event_lines:
            return
        payload = "\n".join(event_lines)
        event_lines.clear()

        if payload == "[DONE]":
            return

        if payload.startswith("[SOURCES]"):
            try:
                sources.update(json.loads(payload[9:]).get("sources", {}))
            except (json.JSONDecodeError, KeyError):
                pass
            return

        if diagnostic_count < _DIAGNOSTIC_CHUNK_LIMIT:
            logger.info("SSE chunk[%d]: %r", diagnostic_count, payload)
            diagnostic_count += 1

        full = assembler.append(payload)
        if on_token:
            on_token(full, payload)

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue

        # Blank line = end of SSE event
        if raw_line == "":
            flush_event()
            continue

        parsed = parse_sse_data_line(raw_line)
        if parsed is None:
            continue

        event_lines.append(parsed)

    flush_event()

    return assembler.finalize(), sources


def stream_chat(
    session_id: str,
    query: str,
    on_token: Callable[[str, str], None] | None = None,
) -> tuple[str, dict]:
    """Stream a chat response from the backend API."""
    payload = {"session_id": session_id, "query": query}
    response = requests.post(
        f"{API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=TIMEOUT_STREAM,
    )

    if response.status_code != 200:
        error_text = response.text
        try:
            detail = response.json().get("detail", error_text)
        except Exception:
            detail = error_text
        raise RuntimeError(detail)

    return consume_chat_stream(response, on_token=on_token)
