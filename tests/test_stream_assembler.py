"""Automated tests for stream assembly and SSE whitespace integrity."""

import unittest

from shared.sse_codec import encode_sse_payload, parse_sse_data_line
from shared.stream_assembler import StreamAssembler
from ui.utils.stream import consume_chat_stream


class TestStreamAssembler(unittest.TestCase):
    """Verify verbatim chunk concatenation."""

    def _assemble(self, chunks: list[str]) -> str:
        asm = StreamAssembler()
        for chunk in chunks:
            asm.append(chunk)
        return asm.finalize()

    def test_trailing_space_chunks(self):
        chunks = ["The ", "document ", "provides "]
        self.assertEqual(self._assemble(chunks), "The document provides ")

    def test_leading_space_chunks(self):
        chunks = ["The", " document", " provides"]
        self.assertEqual(self._assemble(chunks), "The document provides")

    def test_subword_chunks_no_extra_spaces(self):
        chunks = ["carn", "iv", "ores"]
        self.assertEqual(self._assemble(chunks), "carnivores")

    def test_mixed_whitespace(self):
        chunks = ["The ", "document", " provides", " a", " comprehensive", " overview", "."]
        self.assertEqual(
            self._assemble(chunks),
            "The document provides a comprehensive overview.",
        )

    def test_markdown_preserved(self):
        chunks = ["## Heading", "\n", "- Item 1", "\n", "- Item 2"]
        self.assertEqual(self._assemble(chunks), "## Heading\n- Item 1\n- Item 2")

    def test_code_block_preserved(self):
        chunks = ["```python\n", "print(", "'hello'", ")\n", "```"]
        self.assertEqual(self._assemble(chunks), "```python\nprint('hello')\n```")

    def test_empty_chunks_ignored(self):
        asm = StreamAssembler()
        asm.append("Hello")
        asm.append("")
        asm.append(" world")
        self.assertEqual(asm.finalize(), "Hello world")

    def test_no_strip_on_finalize(self):
        asm = StreamAssembler()
        asm.append("  padded  ")
        self.assertEqual(asm.finalize(), "  padded  ")

    def test_lions_sentence(self):
        chunks = [
            "The ", "document ", "provides ", "a ", "comprehensive ",
            "overview ", "of ", "lions", ".",
        ]
        self.assertEqual(
            self._assemble(chunks),
            "The document provides a comprehensive overview of lions.",
        )


class TestSSECodec(unittest.TestCase):
    """Verify SSE encoding/decoding preserves whitespace."""

    def test_trailing_space_survives_roundtrip(self):
        token = "The "
        line = encode_sse_payload(token).split("\n")[0]
        self.assertEqual(parse_sse_data_line(line), "The ")

    def test_leading_space_survives_roundtrip(self):
        token = " document"
        line = encode_sse_payload(token).split("\n")[0]
        self.assertEqual(parse_sse_data_line(line), " document")

    def test_old_strip_would_destroy_spaces(self):
        """Document the bug: str.strip() on payloads removes whitespace."""
        token = "The "
        line = f"data: {token}"
        self.assertEqual(parse_sse_data_line(line), "The ")
        self.assertNotEqual(line[5:].strip(), parse_sse_data_line(line))

    def test_leading_space_old_strip_bug(self):
        token = " document"
        line = f"data: {token}"
        self.assertEqual(parse_sse_data_line(line), " document")
        self.assertEqual(line[5:].strip(), "document")

    def test_multiline_payload(self):
        encoded = encode_sse_payload("line1\nline2")
        lines = [line for line in encoded.split("\n") if line.startswith("data:")]
        payloads = [parse_sse_data_line(line) for line in lines]
        self.assertEqual("\n".join(payloads), "line1\nline2")


class TestStreamConsumer(unittest.TestCase):
    """Integration test for the frontend SSE consumer."""

    def _make_response(self, events: list[str]) -> object:
        body = "".join(events)

        class MockResponse:
            def iter_lines(self, decode_unicode=True):
                for line in body.split("\n"):
                    yield line

        return MockResponse()

    def test_full_sentence_via_sse(self):
        events = [
            encode_sse_payload("The "),
            encode_sse_payload("document "),
            encode_sse_payload("provides "),
            encode_sse_payload("a comprehensive overview of lions."),
            encode_sse_payload("[DONE]"),
        ]
        text, _ = consume_chat_stream(self._make_response(events))
        self.assertEqual(
            text,
            "The document provides a comprehensive overview of lions.",
        )

    def test_leading_space_tokens_via_sse(self):
        events = [
            encode_sse_payload("The"),
            encode_sse_payload(" document"),
            encode_sse_payload(" provides"),
            encode_sse_payload("[DONE]"),
        ]
        text, _ = consume_chat_stream(self._make_response(events))
        self.assertEqual(text, "The document provides")

    def test_subword_tokens_via_sse(self):
        events = [
            encode_sse_payload("carn"),
            encode_sse_payload("iv"),
            encode_sse_payload("ores"),
            encode_sse_payload("[DONE]"),
        ]
        text, _ = consume_chat_stream(self._make_response(events))
        self.assertEqual(text, "carnivores")


if __name__ == "__main__":
    unittest.main()
