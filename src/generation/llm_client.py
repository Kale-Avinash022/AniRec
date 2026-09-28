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
        markers = [
            "Head-to-Head Comparison Candidates:",
            "Comparison Target Shows:",
            "Candidate Shows:",
        ]
        start_bracket = -1
        for marker in markers:
            if marker in prompt:
                idx = prompt.index(marker) + len(marker)
                start_bracket = prompt.find("[", idx)
                break

        if start_bracket == -1:
            start_bracket = prompt.find("[")

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
    """Synthesize a rich, comprehensive 100-150 word description and match rationale."""
    synopsis = candidate.get("synopsis", "") or ""
    tags_raw = candidate.get("tags", "") or ""
    genres_raw = candidate.get("genres", "") or ""
    score = candidate.get("score") or candidate.get("average_score") or 0
    fmt = candidate.get("format", "TV")
    title = candidate.get("title") or candidate.get("title_romaji") or "This anime"

    import re

    cleaned_tags = []
    for t in tags_raw.split(","):
        cleaned = re.sub(r"\s*\(\d+\)", "", t).strip()
        if cleaned and cleaned.lower() not in [x.lower() for x in cleaned_tags]:
            cleaned_tags.append(cleaned)

    genre_list = [g.strip() for g in genres_raw.split(",") if g.strip()]
    key_themes = cleaned_tags[:4] or genre_list[:4] or ["compelling storytelling", "rich character dynamics"]
    theme_str = ", ".join(key_themes)

    clean_syn = _extract_clean_synopsis(synopsis)
    clean_syn = re.sub(r"<[^>]+>", "", clean_syn).strip()

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_syn)
        if len(s.strip()) > 15 and not s.strip().startswith(("http", "(", "Genres:", "Tags:", "Score:"))
    ]

    premise_parts: list[str] = []
    word_count = 0
    for s in sentences:
        words_in_s = len(s.split())
        if word_count + words_in_s <= 85:
            premise_parts.append(s)
            word_count += words_in_s
        else:
            needed = 85 - word_count
            if needed > 6:
                clipped = " ".join(s.split()[:needed])
                premise_parts.append(clipped + "...")
            break

    premise_text = " ".join(premise_parts)
    if not premise_text:
        premise_text = f"{title} is an acclaimed {fmt} production that invites viewers into a meticulously constructed universe full of gripping conflicts and memorable moments."

    if not premise_text.endswith((".", "!", "?", "…", "...")):
        premise_text += "."

    score_str = f"With an impressive community rating of {score}/100," if score > 0 else "Highly regarded by fans,"
    commentary = (
        f"{score_str} it distinguishes itself through its exceptional focus on {theme_str} themes, "
        f"offering emotional resonance, sharp thematic depth, and dynamic pacing that makes every episode compelling."
    )

    combined = f"{premise_text} {commentary.strip()}".strip()
    if len(combined.split()) < 100:
        lead_theme = key_themes[0] if key_themes else "adventure"
        second_theme = key_themes[1] if len(key_themes) > 1 else "thematic"
        elaboration = (
            f" Viewers will appreciate the nuanced character progression, high-stakes storytelling, and memorable "
            f"worldbuilding that elevate its {lead_theme} and {second_theme} motifs into a truly captivating watch."
        )
        combined = f"{combined} {elaboration.strip()}".strip()

    return combined


def _format_comparison_response(candidates: list[dict[str, Any]], user_query: str = "") -> str:
    """Format candidate shows into a structured, analytical head-to-head comparison."""
    if len(candidates) < 2:
        return _format_markdown_response(candidates, user_query=user_query)

    c1, c2 = candidates[0], candidates[1]
    t1 = c1.get("title") or c1.get("title_romaji") or f"Anime #{c1.get('anilist_id', 1)}"
    t2 = c2.get("title") or c2.get("title_romaji") or f"Anime #{c2.get('anilist_id', 2)}"
    y1 = f" ({c1.get('year')})" if c1.get("year") else ""
    y2 = f" ({c2.get('year')})" if c2.get("year") else ""
    s1 = c1.get("score") or c1.get("average_score") or 0
    s2 = c2.get("score") or c2.get("average_score") or 0

    tags1 = [t.strip() for t in (c1.get("tags") or c1.get("genres") or "").split(",") if t.strip()][:4]
    tags2 = [t.strip() for t in (c2.get("tags") or c2.get("genres") or "").split(",") if t.strip()][:4]
    theme1_str = ", ".join(tags1) if tags1 else "action, adventure"
    theme2_str = ", ".join(tags2) if tags2 else "action, fantasy"

    syn1 = _extract_clean_synopsis(c1.get("synopsis", ""))
    syn2 = _extract_clean_synopsis(c2.get("synopsis", ""))
    if len(syn1) > 280:
        syn1 = syn1[:280].rsplit(" ", 1)[0] + "..."
    if len(syn2) > 280:
        syn2 = syn2[:280].rsplit(" ", 1)[0] + "..."

    return (
        f"### ⚔️ **Head-to-Head Comparison: {t1} vs {t2}**\n\n"
        f"#### 📖 **Premise & Worldbuilding**\n"
        f"- **{t1}**{y1}: {syn1}\n"
        f"- **{t2}**{y2}: {syn2}\n\n"
        f"#### 🎭 **Tone, Themes & Atmosphere**\n"
        f"- **{t1}**: Distinctly leans into **{theme1_str}** themes with fast-paced storytelling and intricate world mechanics.\n"
        f"- **{t2}**: Focuses deeply on **{theme2_str}**, delivering emotional resonance, ensemble camaraderie, and high-energy showdowns.\n\n"
        f"#### 👥 **Character Dynamics & Narrative Focus**\n"
        f"- **{t1}** centers on an ambitious protagonist navigating expansive adventures alongside loyal companions.\n"
        f"- **{t2}** emphasizes deep bonds, dramatic character transformations, and escalating rivalries.\n\n"
        f"#### 🏆 **Target Audience & Verdict**\n"
        f"- **Watch {t1} if**: You're looking for modern narrative pacing, creative universe rules, and vibrant action (Rating: ⭐{s1}/100).\n"
        f"- **Watch {t2} if**: You prefer classic long-arc adventure, heartfelt emotional stakes, and explosive team battles (Rating: ⭐{s2}/100).\n"
        f"- **Bottom Line**: Both series offer exceptional experiences for different moods. Pick **{t1}** for fresh thematic twists, or **{t2}** for quintessential epic adventure!"
    )


def _format_markdown_response(candidates: list[dict[str, Any]], user_query: str = "") -> str:
    """Format candidate shows into a rich, engaging Markdown chat response.

    - Comparison requests (Show A vs Show B, compare X and Y): Head-to-head analytical breakdown.
    - Specific anime QA inquiries: Comprehensive narrative paragraphs.
    - Recommendation requests (top N, best N, etc.): Clean numbered list `1. Title — reason`.
    """
    if not candidates:
        return (
            "I searched our catalog based on your request, but couldn't find exact matches.\n\n"
            "Could you try adjusting your filters or describing what themes you'd like to explore?"
        )

    query_lower = user_query.lower()
    is_comparison = (
        any(kw in query_lower for kw in [" vs ", " vs. ", " versus ", "compare ", "difference between", "differences between", "better than", "vs"])
        or (len(candidates) == 2 and " or " in query_lower)
    )

    if is_comparison and len(candidates) >= 2:
        return _format_comparison_response(candidates, user_query=user_query)

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
    for marker in ["User Message:", "User Comparison Request:", "User Request:"]:
        if marker in prompt:
            rest = prompt.split(marker, 1)[1]
            if "\n\nPlease synthesize" in rest:
                return rest.split("\n\nPlease synthesize", 1)[0].strip()
            if "\n\nCandidate Shows:" in rest:
                return rest.split("\n\nCandidate Shows:", 1)[0].strip()
            return rest.strip()
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
