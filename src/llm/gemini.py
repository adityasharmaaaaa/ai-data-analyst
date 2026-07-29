"""
Gemini LLM access via LangChain.

Single place that constructs ChatGoogleGenerativeAI instances so model
name/temperature/API key handling lives in one spot. get_llm() is cached
so repeated calls within a session reuse the same client.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import SETTINGS


class MissingAPIKeyError(RuntimeError):
    """Raised when no Gemini API key is configured."""


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    if not SETTINGS.google_api_key:
        raise MissingAPIKeyError(
            "No Gemini API key found. Set GOOGLE_API_KEY (or GEMINI_API_KEY) "
            "in your .env file - see .env.example."
        )
    return ChatGoogleGenerativeAI(
        model=SETTINGS.gemini_model,
        google_api_key=SETTINGS.google_api_key,
        temperature=temperature if temperature is not None else SETTINGS.gemini_temperature,
    )


def extract_text(content) -> str:
    """
    Normalize an LLM response's `.content` into a plain string.

    Newer LangChain chat models (including Gemini via
    langchain-google-genai) can return `.content` as either a plain
    string OR a list of content blocks, e.g.
    ``[{"type": "text", "text": "...", "extras": {"signature": ...}}]``
    - the list form shows up for multi-part / thinking-capable model
    responses. Every place in this app that displays an answer expects
    plain text, so route `.content` through this instead of assuming
    it's always a str (passing a list straight to st.markdown() renders
    its raw Python repr, which is exactly the wrong thing to show a user).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            block_type = block.get("type")
            if block_type in ("thinking", "reasoning", "redacted_thinking"):
                continue  # internal reasoning, not meant for the user
            text = block.get("text")
            parts.append(str(text) if text is not None else str(block))
        return "\n".join(p for p in parts if p)
    return str(content)
