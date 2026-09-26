"""Grounded System Prompt Engineering & Citation Design (Step 33).

Provides prompts and context formatters that instruct the LLM to:
1. Ground recommendations ONLY in the retrieved candidate list.
2. Explicitly cite matching metadata attributes (community tags, genres, scores, studio).
3. Follow strict anti-spoiler policies (premise-level descriptions only).
4. Output structured JSON adhering to the recommendation contract.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_RECOMMENDER_PROMPT = """\
You are AniRec, an expert anime recommendation engine.

You will receive:
1. The user's query / taste request.
2. A JSON list of retrieved candidate anime with attributes (genres, ranked tags, release year, score, synopsis).

STRICT OPERATIONAL RULES:
1. Candidate Grounding: Recommend ONLY from the provided candidate list. Never invent or recommend shows outside this list.
2. Provenance & Citations: For each recommendation, provide a concrete 2-3 sentence justification citing specific matching attributes (e.g. community tags, thematic tone, genres, score) that satisfy the user's prompt.
3. Anti-Spoiler Policy: Describe only premise-level setup. Never reveal plot twists, character deaths, or climax events.
4. Fallback: If no candidates closely match the user's intent, state that clearly in "reasoning_summary" while providing the closest available options from the candidates.
5. JSON Output Contract: You must output ONLY valid JSON matching this schema:
{
  "recommendations": [
    {
      "anilist_id": 16498,
      "title": "Title Romaji or English",
      "year": 2013,
      "why": "2-3 sentence grounded justification citing specific tags and premise elements.",
      "matched_attributes": ["military (88)", "dark fantasy (77)", "survival (75)"]
    }
  ],
  "reasoning_summary": "Brief 1-2 sentence overview of why these selections match the user's request."
}
"""

SYSTEM_CHAT_PROMPT = """\
You are an anime recommendation assistant.

Use the retrieved anime information as your knowledge source.

RULES:
- Answer the user's question directly.
- Separate normal QA from recommendation queries:
  * For "top N", "best N", or "recommend N" requests, return exactly N recommendations.
  * If asked about a particular anime, provide a comprehensive, well-structured answer in clear narrative paragraphs.
- Each recommendation must be on a separate numbered line in strict ranking order:
  1. <Anime Title> — <brief reason>
  2. <Anime Title> — <brief reason>
  ...
  N. <Anime Title> — <brief reason>
- Never return the recommendations as a paragraph. Do NOT merge multiple recommendations into one paragraph.
- Never expose RAG document IDs, chunk IDs, vector IDs, or internal metadata unless the user explicitly asks for them.
- Do not invent anime information that is not supported by the retrieved context.
"""


def format_candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract and format essential metadata from a retrieved candidate."""
    meta = candidate.get("metadata", {})
    sim = candidate.get("similarity_score", candidate.get("similarity", 0.0))
    rerank = candidate.get("rerank_score")
    return {
        "anilist_id": meta.get("anilist_id", meta.get("anime_id", candidate.get("id"))),
        "title": meta.get("title_english") or meta.get("title_romaji") or f"Anime #{candidate.get('id')}",
        "title_romaji": meta.get("title_romaji", ""),
        "year": meta.get("season_year", 0),
        "format": meta.get("format", "TV"),
        "score": meta.get("average_score", 0),
        "genres": meta.get("genres", ""),
        "tags": meta.get("top_tags", meta.get("top_tags_str", "")),
        "synopsis": candidate.get("document", meta.get("description_clean", "")),
        "similarity": round(float(sim), 4),
        "similarity_score": round(float(sim), 4),
        "rerank_score": round(float(rerank), 4) if rerank is not None else None,
    }


def format_user_prompt(user_query: str, candidates: list[dict[str, Any]]) -> str:
    """Format the candidate list and user query into the grounded prompt payload.

    Args:
        user_query: Raw user request.
        candidates: Filtered & reranked candidates from the retrieval layer.

    Returns:
        Structured string containing user prompt and candidate JSON.
    """
    formatted_candidates = [format_candidate_context(c) for c in candidates]
    candidates_json = json.dumps(formatted_candidates, indent=2, ensure_ascii=False)

    return f"User Request: {user_query}\n\nCandidate Shows:\n{candidates_json}"
