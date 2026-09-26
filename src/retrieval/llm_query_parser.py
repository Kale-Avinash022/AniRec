"""Query Understanding — Structured LLM Filter Extraction (Step 26).

Optional LLM-powered upgrade to the rules-based parser (Step 25).
Handles complex linguistic constraints the regex layer can't reliably parse:

  - "90s anime"                    → rules OK; LLM: year_min=1990, year_max=1999
  - "short 12-episode shows"       → rules miss this; LLM: max_episodes=12
  - "dark psychological without gore" → rules miss negation; LLM: exclude_tags=["gore"]
  - "something like Attack on Titan but lighter"

The LLM returns a Pydantic-validated JSON object via OpenAI's structured output
(``response_format=LLMExtractedQuery``), which guarantees field types.

Fallback: if the LLM call fails (network, quota, parse error), this module
silently returns ``None`` and the caller falls back to the rules-based parser.

Usage:
    # Async (used in FastAPI route context)
    result = await parse_query_with_llm("90s mecha with political themes")
    if result is None:
        result = parse_query_rules(query)  # fallback

    # Sync wrapper (used in CLI / notebook context)
    result = parse_query_with_llm_sync(query)
"""

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logger import logger
from src.retrieval.query_parser import ExtractedFilters, ParsedQuery

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """\
You are a query parsing assistant for an anime recommendation engine.
Given a user's natural language request, extract:
  1. The core semantic/thematic description (mood, tone, themes) — used for vector search.
  2. Any hard metadata constraints (year range, format, score threshold, genres).

Return ONLY the JSON matching the exact schema provided.
Be conservative: only populate a field if the user clearly intends that constraint.
Do not hallucinate constraints that are not stated or strongly implied.
"""


class LLMExtractedQuery(BaseModel):
    """Pydantic schema returned by the LLM structured extraction call."""

    semantic_mood: str = Field(
        description=(
            "The core thematic/atmospheric description of desired anime, "
            "stripped of all metadata constraints. Used directly as embedding query text."
        )
    )
    year_min: Optional[int] = Field(
        None,
        description="Minimum release year if specified (e.g. 2015 for '2015 or later').",
    )
    year_max: Optional[int] = Field(
        None,
        description="Maximum release year if specified (e.g. 1999 for '90s anime').",
    )
    format: Optional[str] = Field(
        None,
        description="Media format if specified: TV, MOVIE, OVA, ONA, or SPECIAL.",
    )
    min_score: Optional[int] = Field(
        None,
        description="Minimum average score out of 100 if specified.",
    )
    genres: list[str] = Field(
        default_factory=list,
        description="List of AniList genres explicitly requested by the user.",
    )


# ---------------------------------------------------------------------------
# Async extractor
# ---------------------------------------------------------------------------


async def parse_query_with_llm(query: str) -> ParsedQuery | None:
    """Extract structured filters from a query using GPT-4o structured output.

    Args:
        query: Raw user query string.

    Returns:
        :class:`ParsedQuery` if the LLM call succeeds, or ``None`` on failure.
        Callers should fall back to :func:`~src.retrieval.query_parser.parse_query_rules`.
    """
    if not settings.OPENAI_API_KEY:
        logger.debug("LLM query parser skipped: OPENAI_API_KEY not set.")
        return None

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        response = await client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format=LLMExtractedQuery,
            timeout=8.0,  # fast timeout so LLM never blocks the pipeline
        )

        parsed: LLMExtractedQuery = response.choices[0].message.parsed
        if parsed is None:
            logger.warning("LLM returned null parsed response; falling back.")
            return None

        filters = ExtractedFilters(
            year_min=parsed.year_min,
            year_max=parsed.year_max,
            format=parsed.format,
            min_score=parsed.min_score,
            genres=parsed.genres,
        )

        logger.debug(
            f"LLM query extraction: semantic='{parsed.semantic_mood}' "
            f"filters={filters.model_dump(exclude_none=True)}"
        )

        return ParsedQuery(
            raw_query=query,
            semantic_text=parsed.semantic_mood,
            filters=filters,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LLM query parser failed ({type(exc).__name__}): {exc} — falling back to rules.")
        return None


# ---------------------------------------------------------------------------
# Sync wrapper (for CLI / testing contexts)
# ---------------------------------------------------------------------------


def parse_query_with_llm_sync(query: str) -> ParsedQuery | None:
    """Synchronous wrapper around :func:`parse_query_with_llm`.

    Uses ``asyncio.run()`` — do NOT call from inside an existing event loop
    (e.g. FastAPI route handlers); use the async version there.
    """
    return asyncio.run(parse_query_with_llm(query))


# ---------------------------------------------------------------------------
# Unified parser — tries LLM first, falls back to rules
# ---------------------------------------------------------------------------


async def parse_query(query: str, prefer_llm: bool = True) -> ParsedQuery:
    """Unified query parser: LLM-first with rules-based fallback.

    Args:
        query:      Raw user query string.
        prefer_llm: If True (default), attempt LLM extraction first.
                    If False, always use the rules-based parser.

    Returns:
        :class:`ParsedQuery` — always succeeds (rules parser never raises).
    """
    from src.retrieval.query_parser import parse_query_rules

    if prefer_llm:
        llm_result = await parse_query_with_llm(query)
        if llm_result is not None:
            return llm_result

    return parse_query_rules(query)
