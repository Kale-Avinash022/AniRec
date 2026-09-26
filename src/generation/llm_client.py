"""Abstract LLM Provider Interface & Implementations (Step 32).

Encapsulates LLM providers (OpenAI, local Ollama, and Mock) behind a uniform
asynchronous interface supporting:
  1. One-shot JSON-mode generation (with cost tracking integration)
  2. Token-by-token asynchronous streaming (for FastAPI SSE / Streamlit)

Usage::

    from src.generation.llm_client import get_chat_model

    llm = get_chat_model()
    # One-shot JSON generation
    response = await llm.generate(prompt="...", system_prompt="...", json_mode=True)

    # Token streaming
    async for token in llm.generate_stream(prompt="...", system_prompt="..."):
        print(token, end="")
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from src.core.config import settings
from src.core.interfaces import BaseChatModel
from src.core.logger import logger


class OpenAIChatModel(BaseChatModel):
    """OpenAI asynchronous chat model adapter (e.g. gpt-4o-mini)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return content

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class OllamaChatModel(BaseChatModel):
    """Local Ollama asynchronous chat model adapter (e.g. llama3.1:8b)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        import httpx

        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        async with self.client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token


def _extract_candidates_from_prompt(prompt: str) -> list[dict[str, Any]]:
    """Extract candidate shows from prompt if available."""
    try:
        marker = "Candidate Shows:"
        if marker in prompt:
            idx = prompt.index(marker) + len(marker)
            start_bracket = prompt.find("[", idx)
            if start_bracket != -1:
                end_bracket = prompt.rfind("]")
                if end_bracket != -1 and end_bracket > start_bracket:
                    json_str = prompt[start_bracket : end_bracket + 1]
                    return json.loads(json_str)
    except Exception:
        pass
    return []


def _extract_requested_count(query: str, default_count: int = 5) -> int:
    """Extract requested number of recommendations if user specifies 'top N', 'best N', 'N anime', etc."""
    import re

    match = re.search(r"\b(?:top|best|recommend|give me|show me)\s+(\d{1,2})\b", query.lower())
    if match:
        try:
            val = int(match.group(1))
            if 1 <= val <= 20:
                return val
        except ValueError:
            pass
    match_anime = re.search(r"\b(\d{1,2})\s+(?:anime|shows|recommendations)\b", query.lower())
    if match_anime:
        try:
            val = int(match_anime.group(1))
            if 1 <= val <= 20:
                return val
        except ValueError:
            pass
    return default_count


def _extract_clean_synopsis(raw_text: str) -> str:
    """Extract clean plot synopsis from raw document text or metadata description."""
    if not raw_text:
        return ""
    if "Description:" in raw_text:
        desc_part = raw_text.split("Description:", 1)[1]
        if "Average score:" in desc_part:
            desc_part = desc_part.split("Average score:", 1)[0]
        return desc_part.strip()
    return raw_text.strip()


def _synthesize_grounded_reason(candidate: dict[str, Any], user_query: str = "") -> str:
    """Synthesize a dynamic, context-aware reason using the actual anime synopsis and attributes."""
    synopsis = candidate.get("synopsis", "") or ""
    tags_raw = candidate.get("tags", "") or ""
    genres_raw = candidate.get("genres", "") or ""
    score = candidate.get("score") or 0
    fmt = candidate.get("format", "TV")

    # Extract top tag names without score numbers (e.g. "military (88)" -> "military")
    import re
    cleaned_tags = []
    for t in tags_raw.split(","):
        cleaned = re.sub(r"\s*\(\d+\)", "", t).strip()
        if cleaned and cleaned.lower() not in [x.lower() for x in cleaned_tags]:
            cleaned_tags.append(cleaned)

    genre_list = [g.strip() for g in genres_raw.split(",") if g.strip()]
    key_themes = (cleaned_tags[:3] or genre_list[:3] or ["compelling storytelling"])
    theme_str = ", ".join(key_themes)

    # Clean synopsis to extract core premise hook
    clean_syn = _extract_clean_synopsis(synopsis)

    # Get first 1-2 clean sentences of plot
    sentences = re.split(r"(?<=[.!?])\s+", clean_syn)
    premise_hook = ""
    for s in sentences:
        s_clean = s.strip()
        if (
            len(s_clean) > 20
            and not s_clean.startswith("http")
            and not s_clean.startswith("(")
            and not s_clean.startswith("Genres:")
            and not s_clean.startswith("Tags:")
        ):
            premise_hook = s_clean
            break

    if len(premise_hook) > 150:
        premise_hook = premise_hook[:147].rsplit(" ", 1)[0] + "..."

    if premise_hook and not premise_hook.endswith((".", "!", "?", "…", "...")):
        premise_hook += "."

    if premise_hook:
        return f"{premise_hook} Features compelling {theme_str} themes with high narrative stakes."
    else:
        score_str = f" (rated {score}/100)" if score > 0 else ""
        return f"A distinguished {fmt} production{score_str} renowned for its rich {theme_str} atmosphere and character depth."


def _format_markdown_response(candidates: list[dict[str, Any]], user_query: str = "") -> str:
    """Format candidate shows into a rich, engaging Markdown chat response.

    - Recommendation requests (top N, best N, etc.): Clean numbered list `1. Title — reason`.
    - Specific anime QA inquiries: Comprehensive narrative paragraphs.
    """
    if not candidates:
        return (
            "I searched our catalog based on your request, but couldn't find exact matches.\n\n"
            "Could you try adjusting your filters or describing what themes you'd like to explore?"
        )

    query_lower = user_query.lower()
    is_specific_qa = any(
        kw in query_lower
        for kw in ["tell me about", "what is", "explain the plot", "synopsis of", "details of", "summary of", "more about", "who is"]
    ) or (len(candidates) == 1 and not any(kw in query_lower for kw in ["recommend", "top", "best", "suggest", "find", "similar"]))

    # ── 1. Specific Anime QA (Paragraph Format) ─────────────────────────────
    if is_specific_qa and len(candidates) >= 1:
        c = candidates[0]
        title = c.get("title") or c.get("title_romaji") or f"Anime #{c.get('anilist_id')}"
        year = c.get("year")
        year_str = f" ({year})" if year else ""
        fmt = c.get("format", "TV")
        genres = c.get("genres", "")
        tags = c.get("tags", "")

        matched_themes = [t.strip() for t in (tags or genres).split(",")[:5] if t.strip()]
        theme_str = ", ".join(matched_themes) if matched_themes else "Compelling Storyline"
        synopsis = c.get("synopsis", "No detailed synopsis available.")

        return (
            f"### 🎬 **{title}**{year_str}\n\n"
            f"**Format**: {fmt} | **Genres & Themes**: {theme_str}\n\n"
            f"**Overview & Story Setup**:\n"
            f"{synopsis}\n\n"
            f"**Themes & Key Highlights**:\n"
            f"This production stands out for its immersive focus on {theme_str.lower()}. "
            f"It delivers high narrative tension, distinctive character dynamics, and memorable worldbuilding that make it a compelling watch.\n\n"
            f"💡 *Would you like recommendations for other anime with a similar tone or aesthetic?*"
        )

    # ── 2. Recommendation Requests (Strict Numbered List Format) ────────────
    requested_n = _extract_requested_count(user_query, default_count=min(len(candidates), 5))
    selected_candidates = candidates[:requested_n]

    lines = []
    for i, c in enumerate(selected_candidates, 1):
        title = c.get("title") or c.get("title_romaji") or f"Anime #{c.get('anilist_id')}"
        year = c.get("year")
        year_str = f" ({year})" if year else ""
        reason = _synthesize_grounded_reason(c, user_query)
        lines.append(f"{i}. **{title}**{year_str} — {reason}")

    return "\n\n".join(lines)


def _format_json_response(candidates: list[dict[str, Any]]) -> str:
    """Format candidate shows into structured JSON matching RecommendResponse schema."""
    if not candidates:
        return json.dumps({
            "recommendations": [
                {
                    "anilist_id": 16498,
                    "title": "Shingeki no Kyojin",
                    "year": 2013,
                    "why": "High-stakes military strategy and dark political intrigue.",
                    "matched_attributes": ["military (88)", "dark fantasy (77)", "survival (75)"]
                }
            ],
            "reasoning_summary": "Top military and political fantasy recommendations matching your request."
        })

    recs = []
    for c in candidates[:5]:
        aid = c.get("anilist_id", 16498)
        title = c.get("title") or c.get("title_romaji", "Anime")
        year = c.get("year", 2020)
        tags = [t.strip() for t in c.get("tags", "").split(",")[:3] if t.strip()]
        genres = [g.strip() for g in c.get("genres", "").split(",")[:3] if g.strip()]
        matched = tags or genres or ["Compelling Story"]
        why_reason = _synthesize_grounded_reason(c)
        recs.append({
            "anilist_id": aid,
            "title": title,
            "year": year,
            "why": why_reason,
            "matched_attributes": matched
        })

    return json.dumps({
        "recommendations": recs,
        "reasoning_summary": f"Selected {len(recs)} top grounded recommendations matching your query."
    })


def _extract_user_query(prompt: str) -> str:
    """Extract user query from formatted prompt string."""
    marker = "User Message:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()
    marker_req = "User Request:"
    if marker_req in prompt:
        return prompt.split(marker_req, 1)[1].split("\n\nCandidate Shows:", 1)[0].strip()
    return prompt


class MockChatModel(BaseChatModel):
    """Deterministic mock chat model for offline testing and CI/CD."""

    def __init__(self, response_text: str | None = None) -> None:
        self.response_text = response_text

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
    ) -> str:
        if self.response_text is not None:
            return self.response_text
        candidates = _extract_candidates_from_prompt(prompt)
        user_query = _extract_user_query(prompt)
        if json_mode:
            return _format_json_response(candidates)
        return _format_markdown_response(candidates, user_query=user_query)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        if self.response_text is not None:
            text = self.response_text
        else:
            candidates = _extract_candidates_from_prompt(prompt)
            user_query = _extract_user_query(prompt)
            text = _format_markdown_response(candidates, user_query=user_query)

        import re
        tokens = re.findall(r"\S+|\n+", text)
        for i, t in enumerate(tokens):
            if "\n" in t:
                yield t
            else:
                next_is_newline = (i + 1 < len(tokens)) and ("\n" in tokens[i + 1])
                yield t + ("" if next_is_newline or i == len(tokens) - 1 else " ")


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """Factory function to instantiate the configured LLM client.

    Args:
        provider: "openai" | "ollama" | "mock" (defaults to settings.LLM_PROVIDER).

    Returns:
        Instance conforming to :class:`~src.core.interfaces.BaseChatModel`.
    """
    selected_provider = (provider or settings.LLM_PROVIDER).lower()

    if selected_provider == "openai":
        if not settings.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set; falling back to MockChatModel")
            return MockChatModel()
        return OpenAIChatModel()
    elif selected_provider == "ollama":
        return OllamaChatModel()
    elif selected_provider in ("mock", "test"):
        return MockChatModel()
    else:
        logger.warning(f"Unknown LLM provider '{selected_provider}'; defaulting to MockChatModel")
        return MockChatModel()
