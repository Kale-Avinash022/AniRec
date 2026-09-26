"""Franchise Deduplication & Lineage Collapse Algorithm (Step 30).

Prevents multiple seasons, spin-offs, sequels, or movies of the same
franchise from monopolizing recommendation slots.

Uses the ``franchise_root_id`` assigned during Step 13 (and stored in ChromaDB
metadata in Step 18) to preserve only the highest-scoring entry per franchise
lineage (or up to ``max_per_franchise`` entries).

Usage::

    from src.retrieval.dedup import deduplicate_franchises

    # candidates: list of candidate dicts with "metadata" and "rerank_score" / "similarity"
    unique_candidates = deduplicate_franchises(candidates, max_per_franchise=1)
"""

from __future__ import annotations

from typing import Any

from src.core.logger import logger


def deduplicate_franchises(
    candidates: list[dict[str, Any]],
    max_per_franchise: int = 1,
) -> list[dict[str, Any]]:
    """Filter candidates to enforce a maximum number of entries per franchise.

    Candidates should already be ordered by preference (e.g. descending by
    similarity or cross-encoder score). The highest-ranked entry for each
    franchise will be retained.

    Args:
        candidates:        List of candidate dicts. Each must contain a
                           ``"metadata"`` dict with ``"franchise_root_id"``
                           or ``"anime_id"``.
        max_per_franchise: Maximum number of shows to keep for any single
                           franchise (default: 1).

    Returns:
        Deduplicated list of candidate dicts preserving relative rank order.
    """
    if not candidates:
        return []

    seen_franchises: dict[int, int] = {}
    deduped: list[dict[str, Any]] = []

    for c in candidates:
        meta = c.get("metadata", {})
        root_id = meta.get("franchise_root_id")
        if root_id is None:
            # Fallback to anime_id or candidate id if root_id is missing
            root_id = meta.get("anime_id", c.get("id"))

        # Convert to int if possible for consistency
        try:
            root_id_key = int(root_id)
        except (ValueError, TypeError):
            root_id_key = hash(str(root_id))

        count = seen_franchises.get(root_id_key, 0)
        if count < max_per_franchise:
            deduped.append(c)
            seen_franchises[root_id_key] = count + 1
        else:
            title = meta.get("title_romaji", f"ID:{c.get('id')}")
            logger.debug(
                f"Deduplicating franchise member '{title}' (root_id={root_id_key})"
            )

    logger.debug(
        f"Franchise deduplication: {len(candidates)} candidates → {len(deduped)} unique"
    )
    return deduped
