"""
LLM Service v3
--------------
Production-style OpenRouter integration
Optimized for RAG-based PDF QA
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
# MODELS
# =========================================================

FREE_MODELS = [

    {
        "id": "openai/gpt-4.1-mini",
        "name": "GPT-4.1 Mini",
        "provider": "OpenAI",
        "context": "128k",
        "note": "Best overall quality",
    },

    {
        "id": "deepseek/deepseek-chat-v3-0324:free",
        "name": "DeepSeek V3",
        "provider": "DeepSeek",
        "context": "128k",
        "note": "Strong free model",
    },

    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "name": "Llama 3.3 70B",
        "provider": "Meta",
        "context": "128k",
        "note": "Good reasoning",
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

        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.model = settings.openrouter_model or "openai/gpt-4.1-mini"

        logger.info(
            "LLMService initialized — model=%s, max_tokens=%d",
            self.model,
            settings.max_tokens,
        )

    # =====================================================
    # CONFIG VALIDATION
    # =====================================================

    def _validate_config(self) -> Optional[str]:
        """
        Validate OpenRouter configuration before sending requests.
        Returns a user-friendly error message, or None if config is valid.
        """
        if not self.api_key or not self.api_key.strip():
            return (
                "OpenRouter API key is not configured. "
                "Set OPENROUTER_API_KEY in your .env file."
            )

        if settings.max_tokens < 1:
            return "Invalid max_tokens setting. MAX_TOKENS must be at least 1."

        if settings.max_tokens > 8192:
            logger.warning(
                "max_tokens=%d is very high and may cause credit errors on OpenRouter.",
                settings.max_tokens,
            )

        return None

    @staticmethod
    def _user_facing_openrouter_error(status_code: int, error_text: str) -> str:
        """
        Map OpenRouter HTTP errors to clean user messages.
        Full technical details are logged separately — never shown to users.
        """
        try:
            payload = json.loads(error_text)
            raw_msg = payload.get("error", {}).get("message", error_text)
        except (json.JSONDecodeError, AttributeError, TypeError):
            raw_msg = error_text

        logger.error(
            "OpenRouter HTTP %d — raw response: %s",
            status_code,
            raw_msg,
        )

        if status_code == 402:
            return (
                "OpenRouter account has insufficient credits. "
                "Please reduce max_tokens or add credits."
            )
        if status_code == 401:
            return (
                "OpenRouter API key is invalid or expired. "
                "Check OPENROUTER_API_KEY in your .env file."
            )
        if status_code == 429:
            return "OpenRouter rate limit reached. Please wait a moment and try again."
        if status_code >= 500:
            return "OpenRouter service is temporarily unavailable. Please try again later."

        return f"OpenRouter request failed (HTTP {status_code}). Please try again."

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

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            chunk = doc.page_content.strip()

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
    ):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }
        ]

        # LAST 6 CHAT TURNS
        for turn in chat_history[-6:]:

            messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })

        # USER PROMPT
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
            logger.error("OpenRouter config validation failed: %s", config_error)
            yield f"⚠️ {config_error}"
            return

        model = model or self.model

        context = self._build_context(
            retrieved_docs
        )

        messages = self._build_messages(
            query,
            context,
            chat_history,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://documind.ai",
            "X-Title": "DocuMind AI",
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "stream": True,
        }

        logger.info(
            "OpenRouter request — model=%s, max_tokens=%d",
            model,
            settings.max_tokens,
        )

        try:

            async with httpx.AsyncClient(
                timeout=300.0
            ) as client:

                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as resp:

                    if resp.status_code != 200:

                        error_body = await resp.aread()
                        error_text = error_body.decode(errors="ignore")

                        user_msg = self._user_facing_openrouter_error(
                            resp.status_code,
                            error_text,
                        )
                        yield f"⚠️ {user_msg}"
                        return

                    # STREAM TOKENS — yield content verbatim, never strip
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
                                    "OpenRouter delta[%d]: %r", chunk_idx, content
                                )
                                chunk_idx += 1

                            yield content

                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue

        except httpx.TimeoutException:

            logger.error("OpenRouter request timed out", exc_info=True)
            yield "⚠️ Request timed out. Please try again."

        except httpx.ConnectError as exc:

            logger.error("OpenRouter connection failed: %s", exc, exc_info=True)
            yield "⚠️ Cannot connect to OpenRouter. Check your network connection."

        except httpx.HTTPError as exc:

            logger.error("OpenRouter HTTP error: %s", exc, exc_info=True)
            yield "⚠️ Network error while contacting OpenRouter. Please try again."

        except Exception as exc:

            logger.error("Unexpected LLM error: %s", exc, exc_info=True)
            yield "⚠️ An unexpected error occurred. Please try again."

    # =====================================================
    # MODELS
    # =====================================================

    def get_available_models(self):

        return FREE_MODELS

    def set_model(
        self,
        model_id: str
    ):

        self.model = model_id

        logger.info(
            f"Switched model to: {model_id}"
        )
