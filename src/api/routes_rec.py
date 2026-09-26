"""POST /recommend One-Shot Endpoint Implementation (Step 40).

Orchestrates the complete recommendation RAG pipeline:
  1. Query parsing & explicit UI filter merging (Steps 25/26).
  2. Hybrid vector search with 5-tier relaxation (Steps 27/28).
  3. Cross-Encoder reranking (Step 29).
  4. Franchise deduplication & lineage collapse (Step 30).
  5. Grounded LLM generation, anti-hallucination verification & spoiler guardrails (Steps 32-37).
  6. Provenance audit tracking & structured response assembly.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from src.api.schemas import (
    CandidateProvenance,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)
from src.api.telemetry import PipelineProfiler
from src.core.logger import logger
from src.generation.llm_client import get_chat_model
from src.generation.pipeline import GroundedRecommender
from src.retrieval.dedup import deduplicate_franchises
from src.retrieval.embedder import get_embedder
from src.retrieval.query_parser import ExtractedFilters, parse_query_rules
from src.retrieval.relaxation import search_with_relaxation
from src.retrieval.reranker import Reranker
from src.retrieval.vectorstore import AnimeVectorStore

router = APIRouter(tags=["recommendations"])

# Lazy singleton service holders
_store: AnimeVectorStore | None = None
_embedder: Any | None = None
_reranker: Reranker | None = None
_recommender: GroundedRecommender | None = None


def get_services() -> tuple[AnimeVectorStore, Any, Reranker, GroundedRecommender]:
    """Retrieve or lazily initialize pipeline service singletons."""
    global _store, _embedder, _reranker, _recommender

    if _store is None:
        _store = AnimeVectorStore()
    if _embedder is None:
        _embedder = get_embedder()
    if _reranker is None:
        _reranker = Reranker()
    if _recommender is None:
        _recommender = GroundedRecommender(get_chat_model())

    return _store, _embedder, _reranker, _recommender


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded Anime Recommendations",
)
async def get_recommendations(
    payload: RecommendRequest,
    request: Request,
) -> RecommendResponse:
    """Generate grounded, provenance-backed anime recommendations."""
    req_id = getattr(request.state, "request_id", "req-unknown")
    profiler = PipelineProfiler(request_id=req_id, endpoint="/recommend")

    store, embedder, reranker, recommender = get_services()

    # ── 1. Query Understanding & Filter Merging ──────────────────────────────
    profiler.start_stage("parsing")
    parsed = parse_query_rules(payload.query)

    # Merge explicit UI filters over parsed filters
    merged_genres = list(set(payload.genres + parsed.filters.genres))
    active_filters = ExtractedFilters(
        year_min=payload.year_min if payload.year_min is not None else parsed.filters.year_min,
        year_max=payload.year_max if payload.year_max is not None else parsed.filters.year_max,
        format=payload.format or parsed.filters.format,
        min_score=payload.min_score if payload.min_score is not None else parsed.filters.min_score,
        genres=merged_genres,
    )
    profiler.end_stage("parsing")

    # Dynamic top_k extraction from query or request body (e.g. "top 13")
    target_k = payload.top_k
    import re

    match = re.search(
        r"\b(?:top|best|recommend|give me|show me)\s*(\d{1,2})\b|\b(\d{1,2})\s*(?:anime|shows|recommendations)\b",
        payload.query.lower(),
    )
    if match:
        extracted = int(match.group(1) or match.group(2))
        if 1 <= extracted <= 30:
            target_k = extracted

    retrieval_k = max(target_k * 4, 30)
    rerank_k = max(target_k * 2, 20)

    # ── 2. Vector Search with Progressive Relaxation ─────────────────────────
    profiler.start_stage("retrieval")
    candidates, relaxations = search_with_relaxation(
        store=store,
        embedder=embedder,
        semantic_text=parsed.semantic_text,
        initial_filters=active_filters,
        min_results_needed=min(retrieval_k, 20),
        top_k=retrieval_k,
    )

    # Exclude user-specified IDs (e.g. already watched)
    if payload.exclude_ids:
        exclude_set = set(payload.exclude_ids)

        def _get_id(c: dict[str, Any]) -> int:
            meta = c.get("metadata", {})
            val = meta.get("anime_id") or meta.get("anilist_id") or c.get("id")
            if isinstance(val, int):
                return val
            try:
                return int(str(val).replace("anilist_", "").strip())
            except (ValueError, TypeError):
                return 0

        candidates = [c for c in candidates if _get_id(c) not in exclude_set]
    profiler.end_stage("retrieval")

    if not candidates:
        profiler.finish_and_log()
        logger.warning(f"No candidates found for query: '{payload.query}' [{req_id}]")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No anime candidates found matching your criteria for '{payload.query}'.",
        )

    # ── 3. Cross-Encoder Reranking ───────────────────────────────────────────
    profiler.start_stage("reranking")
    rescored = reranker.rerank(
        query=payload.query,
        candidates=candidates,
        top_k=rerank_k,
    )

    # ── 4. Franchise Deduplication ───────────────────────────────────────────
    deduped = deduplicate_franchises(rescored, max_per_franchise=1)
    top_candidates = deduped[:target_k]
    profiler.end_stage("reranking")

    # ── 5. Grounded Generation & Anti-Hallucination Verification ──────────────
    profiler.start_stage("generation")
    gen_output = await recommender.recommend(
        user_query=payload.query,
        candidates=top_candidates,
    )
    profiler.end_stage("generation")

    # ── 6. Response & Provenance Assembly ────────────────────────────────────
    provenance = [
        CandidateProvenance(
            anilist_id=int(c.get("metadata", {}).get("anilist_id", c.get("id", 0))),
            title=c.get("metadata", {}).get("title_romaji", "Unknown"),
            similarity_score=round(float(c.get("similarity", 0.0)), 4),
            rerank_score=round(float(c.get("rerank_score", 0.0)), 4)
            if c.get("rerank_score") is not None
            else None,
            genres=str(c.get("metadata", {}).get("genres", "")),
            top_tags=str(c.get("metadata", {}).get("top_tags", "")),
        )
        for c in top_candidates
    ]

    meta_lookup = {
        int(c.get("metadata", {}).get("anilist_id", c.get("id", 0))): c.get("metadata", {})
        for c in top_candidates
    }

    items: list[RecommendedItem] = []
    for pick in gen_output.recommendations:
        meta = meta_lookup.get(pick.anilist_id, {})
        items.append(
            RecommendedItem(
                anilist_id=pick.anilist_id,
                title=pick.title,
                year=pick.year or meta.get("season_year"),
                format=meta.get("format"),
                score=meta.get("average_score"),
                why=pick.why,
                matched_attributes=pick.matched_attributes,
                cover_image_url=meta.get("cover_image_url"),
                site_url=meta.get("site_url") or f"https://anilist.co/anime/{pick.anilist_id}",
            )
        )

    telemetry_event = profiler.finish_and_log()
    total_latency_ms = telemetry_event.get("stage_latencies_ms", {}).get("total", 0.0)

    return RecommendResponse(
        request_id=req_id,
        query=payload.query,
        recommendations=items,
        reasoning_summary=gen_output.reasoning_summary,
        retrieved_provenance=provenance,
        relaxations_applied=relaxations,
        latency_ms=total_latency_ms,
    )
