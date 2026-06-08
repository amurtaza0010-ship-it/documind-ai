"""
LLM Service v3 — Groq Edition
------------------------------
Groq API integration (ultra-fast inference)
Optimized for RAG-based PDF QA
Model: llama-3.1-8b-instant (default) — blazing fast & free
"""

import httpx
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_DIAGNOSTIC_CHUNK_LIMIT = 50

# =========================================================
# GROQ CONFIG
# =========================================================

GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
DEFAULT_MODEL  = "llama-3.1-8b-instant"
MAX_TOKENS     = 1024
TEMPERATURE    = 0.1

# =========================================================
# AVAILABLE GROQ MODELS
# =========================================================

GROQ_MODELS = [
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B Instant",
        "provider": "Groq",
        "context": "128k",
        "note": "Fastest — recommended for RAG",
    },
    {
        "id": "llama3-8b-8192",
        "name": "Llama 3 8B",
        "provider": "Groq",
        "context": "8k",
        "note": "Fast and reliable",
    },
    {
        "id": "gemma2-9b-it",
        "name": "Gemma 2 9B",
        "provider": "Groq",
        "context": "8k",
        "note": "Good quality free model",
    },
    {
        "id": "mixtral-8x7b-32768",
        "name": "Mixtral 8x7B",
        "provider": "Groq",
        "context": "32k",
        "note": "Strong reasoning, larger context",
    },
]

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are DocuMind AI, an expert PDF assistant specialized in document question answering.

Core Rules:
1. Answer ONLY using the provided document context.
2. If relevant information exists, answer confidently and directly.
3. Never say information is unavailable if relevant context exists.
4. Never hallucinate or invent facts outside the documents.
5. Keep responses natural, professional, and concise.
6. Mention source document and page numbers whenever possible.
7. Prefer the most relevant information from the context.
8. If no relevant information exists, say:
   "This information is not available in uploaded documents."

Response Style:
- Use clean, well-structured markdown when helpful (headings, bullet lists, bold for emphasis).
- Use paragraphs for narrative text.
- Avoid repetitive phrases.
- Summarize intelligently when context is large.
- Cite source document and page numbers inline when referencing specific facts.
"""

# =========================================================
# LLM SERVICE
# =========================================================

class LLMService:

    def __init__(self):

        self.api_key  = settings.groq_api_key
        self.base_url = GROQ_BASE_URL
        self.model    = DEFAULT_MODEL

        logger.info(
            "LLMService initialized — Groq | model=%s | max_tokens=%d",
            self.model,
            MAX_TOKENS,
        )

    # =====================================================
    # CONFIG VALIDATION
    # =====================================================

    def _validate_config(self) -> Optional[str]:
        """
        Validate Groq config before sending requests.
        Returns error message or None if OK.
        """
        if not self.api_key or not self.api_key.strip():
            return (
                "Groq API key is not configured. "
                "Add GROQ_API_KEY in your .env file."
            )
        return None

    @staticmethod
    def _user_facing_groq_error(status_code: int, error_text: str) -> str:
        """
        Map Groq HTTP errors to clean user messages.
        """
        try:
            payload  = json.loads(error_text)
            raw_msg  = payload.get("error", {}).get("message", error_text)
        except (json.JSONDecodeError, AttributeError, TypeError):
            raw_msg = error_text

        logger.error(
            "Groq HTTP %d — raw response: %s",
            status_code,
            raw_msg,
        )

        if status_code == 401:
            return (
                "Groq API key is invalid or expired. "
                "Check GROQ_API_KEY in your .env file."
            )
        if status_code == 429:
            return "Groq rate limit reached. Please wait a moment and try again."
        if status_code == 413:
            return "Request too large. Try reducing the document context or question length."
        if status_code >= 500:
            return "Groq service is temporarily unavailable. Please try again later."

        return f"Groq request failed (HTTP {status_code}). Please try again."

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def _build_context(
        self,
        retrieved_docs: List[Tuple[Document, float]]
    ) -> str:

        if not retrieved_docs:
            return "No relevant context found."

        context_parts = []

        for i, (doc, score) in enumerate(retrieved_docs, start=1):

            source = doc.metadata.get("source", "Unknown")
            page   = doc.metadata.get("page", "?")
            chunk  = doc.page_content.strip()

            context_parts.append(
                f"""
[Context {i}]
Source: {source}
Page: {page}
Relevance Score: {round(score, 3)}

Content:
{chunk}
"""
            )

        return "\n\n".join(context_parts)

    # =====================================================
    # BUILD MESSAGES
    # =====================================================

    def _build_messages(
        self,
        query: str,
        context: str,
        chat_history: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # Last 6 chat turns
        for turn in chat_history[-6:]:
            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        # User prompt with RAG context
        user_prompt = f"""
Use the following document context to answer the user's question.

Document Context:
{context}

Question:
{query}

Instructions:
- Answer directly from the context.
- If relevant information exists, do NOT deny it.
- Keep the answer natural and well-structured.
- Mention document source and page number.
- Do not hallucinate outside the provided context.
"""

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        return messages

    # =====================================================
    # STREAM RESPONSE
    # =====================================================

    async def stream_response(
        self,
        query: str,
        retrieved_docs: List[Tuple[Document, float]],
        chat_history: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:

        config_error = self._validate_config()
        if config_error:
            logger.error("Groq config validation failed: %s", config_error)
            yield f"⚠️ {config_error}"
            return

        model = model or self.model

        context  = self._build_context(retrieved_docs)
        messages = self._build_messages(query, context, chat_history)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "stream": True,
        }

        logger.info(
            "Groq request — model=%s, max_tokens=%d",
            model,
            MAX_TOKENS,
        )

        try:

            async with httpx.AsyncClient(timeout=60.0) as client:

                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:

                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_text = error_body.decode(errors="ignore")
                        user_msg   = self._user_facing_groq_error(
                            resp.status_code,
                            error_text,
                        )
                        yield f"⚠️ {user_msg}"
                        return

                    # Stream tokens — same SSE format as OpenAI/OpenRouter
                    chunk_idx = 0
                    async for line in resp.aiter_lines():

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)

                            content = (
                                chunk["choices"][0]
                                .get("delta", {})
                                .get("content", "")
                            )

                            if not content:
                                continue

                            if chunk_idx < _DIAGNOSTIC_CHUNK_LIMIT:
                                logger.info(
                                    "Groq delta[%d]: %r", chunk_idx, content
                                )
                                chunk_idx += 1

                            yield content

                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue

        except httpx.TimeoutException:
            logger.error("Groq request timed out", exc_info=True)
            yield "⚠️ Request timed out. Please try again."

        except httpx.ConnectError as exc:
            logger.error("Groq connection failed: %s", exc, exc_info=True)
            yield "⚠️ Cannot connect to Groq. Check your internet connection."

        except httpx.HTTPError as exc:
            logger.error("Groq HTTP error: %s", exc, exc_info=True)
            yield "⚠️ Network error while contacting Groq. Please try again."

        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc, exc_info=True)
            yield "⚠️ An unexpected error occurred. Please try again."

    # =====================================================
    # MODELS
    # =====================================================

    def get_available_models(self):
        return GROQ_MODELS

    def set_model(self, model_id: str):
        self.model = model_id
        logger.info("Switched model to: %s", model_id)