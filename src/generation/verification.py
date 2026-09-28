"""Anti-Hallucination Post-Generation Verification Gate (Step 35).

Acts as a strict post-generation firewall:
1. Validates that every generated show in ``RecommendationOutput`` strictly
   originates from the retrieved candidate pool.
2. Filters out hallucinated or ungrounded titles.
3. Automatically fixes metadata discrepancies (e.g. matching title to correct ID).
4. Logs hallucination violations for faithfulness evaluation metrics.

Usage::

    from src.generation.verification import verify_recommendations_against_candidates

    verified_output, is_faithful = verify_recommendations_against_candidates(
        output=llm_output,
        candidates=retrieved_candidates,
    )
"""

from __future__ import annotations

from typing import Any

from src.core.logger import logger
from src.generation.contract import RecommendationOutput, RecommendedShow


def build_candidate_whitelist(
    candidates: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Build lookup tables from retrieved candidate list for fast verification.

    Returns:
        - ``id_to_meta``: map of int anilist_id -> candidate metadata
        - ``title_to_id``: map of lowercase title (romaji & english) -> int anilist_id
    """
    id_to_meta: dict[int, dict[str, Any]] = {}
    title_to_id: dict[str, int] = {}

    for c in candidates:
        meta = c.get("metadata", {})
        raw_id = meta.get("anilist_id", meta.get("anime_id", c.get("id")))
        try:
            aid = int(raw_id)
        except (TypeError, ValueError):
            continue

        id_to_meta[aid] = meta

        romaji = meta.get("title_romaji")
        if romaji:
            title_to_id[romaji.strip().lower()] = aid

        english = meta.get("title_english")
        if english:
            title_to_id[english.strip().lower()] = aid

    return id_to_meta, title_to_id


def verify_recommendations_against_candidates(
    output: RecommendationOutput,
    candidates: list[dict[str, Any]],
) -> tuple[RecommendationOutput, bool]:
    """Assert that every recommended show exists in the retrieved candidate pool.

    Args:
        output:     :class:`RecommendationOutput` parsed from LLM.
        candidates: List of candidate dicts supplied to the prompt.

    Returns:
        2-tuple:
        - ``output``: Updated RecommendationOutput containing only verified picks.
        - ``is_faithful``: True if 0 hallucinations occurred; False if any were dropped.
    """
    if not candidates:
        logger.warning("Empty candidate pool provided to verification gate.")
        output.recommendations = []
        return output, False

    id_to_meta, title_to_id = build_candidate_whitelist(candidates)

    verified_picks: list[RecommendedShow] = []
    hallucination_occurred = False

    for pick in output.recommendations:
        title_lower = pick.title.strip().lower()

        # Check 1: Exact ID match in candidate whitelist
        if pick.anilist_id in id_to_meta:
            meta = id_to_meta[pick.anilist_id]
            enriched_pick = pick.model_copy(
                update={
                    "cover_image_url": pick.cover_image_url or meta.get("cover_image_url"),
                    "site_url": pick.site_url or meta.get("site_url") or f"https://anilist.co/anime/{pick.anilist_id}",
                    "score": pick.score if pick.score is not None else meta.get("average_score"),
                    "format": pick.format or meta.get("format"),
                }
            )
            verified_picks.append(enriched_pick)
            continue

        # Check 2: Title match in candidate whitelist (correct the ID if mismatched)
        if title_lower in title_to_id:
            matched_id = title_to_id[title_lower]
            meta = id_to_meta.get(matched_id, {})
            logger.info(
                f"Correcting ID for '{pick.title}': was {pick.anilist_id} → {matched_id}"
            )
            corrected_pick = pick.model_copy(
                update={
                    "anilist_id": matched_id,
                    "cover_image_url": pick.cover_image_url or meta.get("cover_image_url"),
                    "site_url": pick.site_url or meta.get("site_url") or f"https://anilist.co/anime/{matched_id}",
                    "score": pick.score if pick.score is not None else meta.get("average_score"),
                    "format": pick.format or meta.get("format"),
                }
            )
            verified_picks.append(corrected_pick)
            continue

        # Check 3: Not in whitelist -> Hallucination violation
        hallucination_occurred = True
        logger.warning(
            f"GUARDRAIL VIOLATION: Dropping hallucinated show: ID={pick.anilist_id}, Title='{pick.title}'. Not in candidate pool!"
        )

    output.recommendations = verified_picks
    is_faithful = not hallucination_occurred

    logger.debug(
        f"Anti-hallucination verification: {len(verified_picks)} verified, faithful={is_faithful}"
    )
    return output, is_faithful
