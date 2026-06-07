"""
Server-Sent Events (SSE) encode/decode for chat token streaming.

CRITICAL: Never call str.strip() on token payloads. Stripping destroys
inter-token whitespace when models emit leading/trailing spaces on chunks.
"""

from __future__ import annotations


def encode_sse_payload(payload: str) -> str:
    """
    Encode a payload as one SSE event (ends with blank line).

    Multi-line payloads are split per SSE spec — each line is a separate
    ``data:`` field, rejoined with newlines on decode.
    """
    if payload == "":
        return "data:\n\n"
    lines = payload.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def parse_sse_data_line(line: str) -> str | None:
    """
    Parse a single ``data:`` line and return the payload.

    Only removes:
      - line-ending CR/LF (via rstrip)
      - the optional single space after ``data:`` per SSE spec

    Never strips leading/trailing whitespace from the actual token content.
    """
    if not line:
        return None

    if isinstance(line, bytes):
        line = line.decode("utf-8")

    # Remove transport line endings only — NOT token whitespace
    line = line.rstrip("\r\n")
    if not line:
        return None

    if not line.startswith("data:"):
        return None

    payload = line[5:]
    # SSE spec: one optional leading space after the colon
    if payload.startswith(" "):
        payload = payload[1:]

    return payload


def is_sse_event_boundary(line: str) -> bool:
    """Return True if this line marks the end of an SSE event (blank line)."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    return line.rstrip("\r\n") == ""
