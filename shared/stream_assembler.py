"""
Stream token assembly — preserves model whitespace exactly.

OpenRouter (and OpenAI-compatible APIs) emit delta.content chunks whose
boundaries are arbitrary tokenizer splits. Whitespace may appear as:
  - trailing space on a chunk:  "The ", "document "
  - leading space on a chunk:    "The", " document", " provides"
  - sub-word pieces:             "carn", "iv", "ores"

The ONLY correct reconstruction is verbatim concatenation. Never inject
spaces, never strip chunk payloads, never collapse whitespace.
"""

from __future__ import annotations


class StreamAssembler:
    """Accumulates streamed text chunks without mutating whitespace."""

    __slots__ = ("_buffer", "chunk_count")

    def __init__(self) -> None:
        self._buffer = ""
        self.chunk_count = 0

    @property
    def text(self) -> str:
        return self._buffer

    def append(self, chunk: str) -> str:
        """
        Append a chunk verbatim and return the full buffer.

        Empty chunks are ignored (OpenRouter sometimes sends empty deltas).
        """
        if not chunk:
            return self._buffer
        self._buffer += chunk
        self.chunk_count += 1
        return self._buffer

    def reset(self) -> None:
        self._buffer = ""
        self.chunk_count = 0

    def finalize(self) -> str:
        """
        Return the assembled text with only invisible Unicode artifacts removed.

        Does NOT strip spaces, newlines, or markdown. Only removes zero-width
        characters that can break rendering but are never intentional content.
        """
        if not self._buffer:
            return self._buffer
        return _remove_zero_width(self._buffer)


def _remove_zero_width(text: str) -> str:
    """Remove zero-width chars only — never touch visible whitespace."""
    for char in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(char, "")
    return text
