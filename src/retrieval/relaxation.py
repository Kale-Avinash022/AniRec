"""Progressive Filter Relaxation Strategy (Step 28).

When users apply overly strict constraints (e.g. "2024+, Psychological, score>=90"),
the strict filtered search may return fewer than ``min_results_needed`` candidates.
Instead of failing silently, the engine relaxes constraints in priority order:

  Tier 1 — Strict:  all filters applied
  Tier 2 — Relax score:  drop min_score (most common over-constraint)
  Tier 3 — Relax year:   drop year_min / year_max
  Tier 4 — Relax genre:  drop genre filters
  Tier 5 — Pure semantic: drop all filters (always returns results)

Each relaxation step logs a human-readable note returned to the caller so the
frontend/LLM can inform the user: "Your score filter was relaxed to find results."

Usage::

    candidates, notes = search_with_relaxation(
        store=store,
        embedder=embedder,
        semantic_text="psychological thriller",
        initial_filters=ExtractedFilters(year_min=2024, min_score=90),
    )
    # → candidates: list of result dicts
    # → notes: ["Relaxed minimum score filter from 90"]
"""

from __future__ import annotations

import copy
from typing import Any

from src.core.logger import logger
from src.retrieval.embedder import BaseEmbedder
from src.retrieval.query_parser import ExtractedFilters
from src.retrieval.search import search_candidates
from src.retrieval.vectorstore import AnimeVectorStore


def search_with_relaxation(
    store: AnimeVectorStore,
    embedder: BaseEmbedder,
    semantic_text: str,
    initial_filters: ExtractedFilters,
    min_results_needed: int = 5,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Execute a hybrid search with automatic filter relaxation on sparse results.

    Args:
        store:               :class:`AnimeVectorStore` instance.
        embedder:            :class:`BaseEmbedder` instance.
        semantic_text:       Cleaned semantic query text.
        initial_filters:     Starting filters from the query parser.
        min_results_needed:  Minimum candidates before triggering relaxation.
        top_k:               Maximum candidates to retrieve per attempt.

    Returns:
        2-tuple of:
        - ``candidates``: List of candidate dicts ordered by similarity.
        - ``relaxation_notes``: Human-readable list of applied relaxations.
    """
    relaxation_notes: list[str] = []
    current = copy.deepcopy(initial_filters)

    # ── Tier 1: Strict search ─────────────────────────────────────────────
    candidates = search_candidates(store, embedder, semantic_text, current, top_k=top_k)
    logger.debug(f"Tier 1 (strict): {len(candidates)} results")
    if len(candidates) >= min_results_needed:
        return candidates, relaxation_notes

    # ── Tier 2: Relax min_score ───────────────────────────────────────────
    if current.min_score is not None:
        note = f"Score filter (≥{current.min_score}) relaxed — showing broader results."
        relaxation_notes.append(note)
        current.min_score = None
        candidates = search_candidates(store, embedder, semantic_text, current, top_k=top_k)
        logger.debug(f"Tier 2 (no score): {len(candidates)} results")
        if len(candidates) >= min_results_needed:
            return candidates, relaxation_notes

    # ── Tier 3: Relax year constraints ────────────────────────────────────
    year_relaxed = False
    if current.year_min is not None:
        note = f"Year filter (from {current.year_min}) relaxed — including older titles."
        relaxation_notes.append(note)
        current.year_min = None
        year_relaxed = True
    if current.year_max is not None:
        note = f"Year filter (up to {current.year_max}) relaxed — including newer titles."
        relaxation_notes.append(note)
        current.year_max = None
        year_relaxed = True

    if year_relaxed:
        candidates = search_candidates(store, embedder, semantic_text, current, top_k=top_k)
        logger.debug(f"Tier 3 (no year): {len(candidates)} results")
        if len(candidates) >= min_results_needed:
            return candidates, relaxation_notes

    # ── Tier 4: Relax genre filters ───────────────────────────────────────
    if current.genres:
        relaxation_notes.append(
            "Genre filter relaxed — showing thematically similar shows from other genres."
        )
        current.genres = []
        candidates = search_candidates(store, embedder, semantic_text, current, top_k=top_k)
        logger.debug(f"Tier 4 (no genre): {len(candidates)} results")
        if len(candidates) >= min_results_needed:
            return candidates, relaxation_notes

    # ── Tier 5: Pure semantic — drop all filters ──────────────────────────
    if not current.is_empty():
        relaxation_notes.append(
            "All filters relaxed — showing best thematic matches across the full catalog."
        )
        candidates = search_candidates(
            store, embedder, semantic_text, ExtractedFilters(), top_k=top_k
        )
        logger.debug(f"Tier 5 (pure semantic): {len(candidates)} results")

    return candidates, relaxation_notes
