"""Generation Self-Correction & Fallback Pipeline (Step 36).

Orchestrates grounded recommendation generation with:
1. Turn 1 generation using the grounded system prompt and structured JSON schema.
2. Self-correction retry upon detecting JSON parse error or hallucination violation.
3. Deterministic template fallback if the LLM provider fails completely.

Usage::

    from src.generation.pipeline import GroundedRecommender
    from src.generation.llm_client import get_chat_model

    recommender = GroundedRecommender(llm=get_chat_model())
    output = await recommender.recommend(
        user_query="dark military fantasy",
        candidates=retrieved_candidates,
    )
"""

from __future__ import annotations

from typing import Any

from src.core.logger import logger
from src.generation.contract import (
    RecommendationOutput,
    RecommendedShow,
    parse_llm_json,
)
from src.generation.llm_client import BaseChatModel, get_chat_model
from src.generation.prompts import (
    SYSTEM_RECOMMENDER_PROMPT,
    format_candidate_context,
    format_user_prompt,
)
from src.generation.verification import verify_recommendations_against_candidates


def build_deterministic_fallback(
    user_query: str,
    candidates: list[dict[str, Any]],
    reason: str = "Fallback template",
) -> RecommendationOutput:
    """Construct a high-quality deterministic recommendation response without an LLM.

    Used when LLM APIs are unreachable, quota-exhausted, or fail validation twice.
    """
    recommendations: list[RecommendedShow] = []

    for c in candidates[:5]:
        ctx = format_candidate_context(c)
        title = ctx["title"]
        year = ctx.get("year")
        genres = ctx.get("genres", "")
        tags = ctx.get("tags", "")

        matched: list[str] = []
        if tags:
            matched.extend([t.strip() for t in tags.split(",")[:3] if t.strip()])
        elif genres:
            matched.extend([g.strip() for g in genres.split(",")[:3] if g.strip()])

        why = (
            f"{title} ({year or 'Classic'}) matches your search with notable themes "
            f"including {', '.join(matched) if matched else 'compelling storytelling'}."
        )

        recommendations.append(
            RecommendedShow(
                anilist_id=ctx["anilist_id"],
                title=title,
                year=year if year else None,
                why=why,
                matched_attributes=matched,
            )
        )

    return RecommendationOutput(
        recommendations=recommendations,
        reasoning_summary=(
            f"Retrieved top-ranked matches for '{user_query}' based on semantic similarity and catalog attributes ({reason})."
        ),
    )


class GroundedRecommender:
    """Two-stage generation engine with anti-hallucination verification and self-correction."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm or get_chat_model()

    async def recommend(
        self,
        user_query: str,
        candidates: list[dict[str, Any]],
        max_retries: int = 1,
    ) -> RecommendationOutput:
        """Generate verified, grounded anime recommendations.

        Args:
            user_query:  User prompt.
            candidates:  Retrieved candidates list from Phase 5 (Step 31).
            max_retries: Number of self-correction attempts upon failure.

        Returns:
            :class:`RecommendationOutput` guaranteed to only contain verified shows.
        """
        if not candidates:
            return RecommendationOutput(
                recommendations=[],
                reasoning_summary="No matching anime found for your criteria in the catalog.",
            )

        user_prompt = format_user_prompt(user_query, candidates)

        # ── Turn 1: Initial Generation ─────────────────────────────────────────
        try:
            raw_response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=SYSTEM_RECOMMENDER_PROMPT,
                json_mode=True,
            )
            parsed_output = parse_llm_json(raw_response)
        except Exception as exc:
            logger.warning(f"Turn 1 generation/parsing failed ({exc}). Retrying once...")
            try:
                raw_response = await self.llm.generate(
                    prompt=user_prompt + "\n\nCRITICAL: Return ONLY valid JSON matching the exact schema.",
                    system_prompt=SYSTEM_RECOMMENDER_PROMPT,
                    json_mode=True,
                )
                parsed_output = parse_llm_json(raw_response)
            except Exception as retry_exc:
                logger.error(f"Turn 1 retry failed ({retry_exc}). Invoking deterministic fallback.")
                return build_deterministic_fallback(user_query, candidates, reason="LLM unavailable")

        # ── Post-Generation Verification Gate ─────────────────────────────────
        verified_output, is_faithful = verify_recommendations_against_candidates(
            output=parsed_output,
            candidates=candidates,
        )

        # ── Turn 2: Self-Correction if Hallucinations were Detected ───────────
        if not is_faithful and max_retries > 0:
            logger.warning("Guardrail violation detected. Executing self-correction prompt...")
            correction_prompt = (
                user_prompt
                + "\n\nCRITICAL ERROR IN PREVIOUS ATTEMPT: You recommended shows that were NOT in the provided candidate list. "
                "You MUST select exclusively from the provided Candidate Shows and cite their exact anilist_id."
            )
            try:
                raw_correction = await self.llm.generate(
                    prompt=correction_prompt,
                    system_prompt=SYSTEM_RECOMMENDER_PROMPT,
                    json_mode=True,
                )
                parsed_corrected = parse_llm_json(raw_correction)
                verified_output, _ = verify_recommendations_against_candidates(
                    output=parsed_corrected,
                    candidates=candidates,
                )
            except Exception as corr_exc:
                logger.warning(f"Self-correction failed ({corr_exc}); using previous verified subset.")

        # If after verification no recommendations remained, use deterministic fallback
        if not verified_output.recommendations:
            logger.info("Verification left 0 recommendations. Using deterministic fallback.")
            return build_deterministic_fallback(user_query, candidates, reason="Verified candidate fallback")

        return verified_output
