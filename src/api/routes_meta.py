"""GET /anime/{id} Show Detail & GET /health Endpoints (Step 42).

Provides:
  - ``GET /health``: Liveness and status check for container orchestrators.
  - ``GET /anime/{id}``: Full metadata and franchise details for UI modals / popovers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api.routes_rec import get_services
from src.api.schemas import AnimeDetailResponse, HealthResponse
from src.core.config import settings
from src.core.logger import logger

router = APIRouter(tags=["metadata"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Document Stats",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> HealthResponse:
    """Return health status, active environment, and ChromaDB vector count."""
    store, _, _, _ = get_services()
    indexed_count = store.count()

    return HealthResponse(
        status="healthy",
        indexed_documents=indexed_count,
        environment=settings.ENVIRONMENT,
        llm_provider=settings.LLM_PROVIDER,
        version="1.0.0",
    )


@router.get(
    "/anime/{anime_id}",
    response_model=AnimeDetailResponse,
    summary="Get Anime Show Detail",
    status_code=status.HTTP_200_OK,
)
async def get_anime_detail(anime_id: int) -> AnimeDetailResponse:
    """Fetch complete show metadata from the ChromaDB collection by AniList ID."""
    store, _, _, _ = get_services()

    # Search for both standard ID format "anilist_{id}" and plain integer string "{id}"
    possible_ids = [f"anilist_{anime_id}", str(anime_id)]
    results = store.collection.get(
        ids=possible_ids,
        include=["metadatas", "documents"],
    )

    ids = results.get("ids", [])
    if not ids or not ids[0]:
        logger.warning(f"Anime ID {anime_id} not found in ChromaDB collection.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anime ID {anime_id} was not found in the catalog.",
        )

    meta = results["metadatas"][0] if results.get("metadatas") else {}
    doc_text = results["documents"][0] if results.get("documents") else ""

    genres_str = meta.get("genres", "")
    genres_list = [g.strip() for g in genres_str.split(",") if g.strip()] if isinstance(genres_str, str) else []

    tags_str = meta.get("top_tags", meta.get("top_tags_str", ""))
    tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if isinstance(tags_str, str) else []

    return AnimeDetailResponse(
        anime_id=anime_id,
        id_mal=meta.get("id_mal"),
        franchise_root_id=meta.get("franchise_root_id", anime_id),
        title_romaji=meta.get("title_romaji", f"Anime #{anime_id}"),
        title_english=meta.get("title_english") or None,
        season_year=meta.get("season_year") or None,
        season=meta.get("season") or None,
        format=meta.get("format", "TV"),
        status=meta.get("status", "FINISHED"),
        episodes=meta.get("episodes") or None,
        duration_mins=meta.get("duration_mins") or None,
        average_score=meta.get("average_score") or None,
        popularity=meta.get("popularity") or None,
        genres=genres_list,
        top_tags=tags_list,
        description_clean=doc_text or meta.get("description_clean", ""),
        cover_image_url=meta.get("cover_image_url") or None,
        site_url=meta.get("site_url") or f"https://anilist.co/anime/{anime_id}",
    )
