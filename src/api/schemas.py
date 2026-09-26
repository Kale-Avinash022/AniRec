"""Pydantic Request/Response API Schemas (Step 39).

Defines strongly-typed boundary contracts for:
  - ``POST /recommend``: One-shot recommendation request & full provenance response
  - ``POST /chat``: Multi-turn chat message payload
  - ``GET /anime/{id}``: Show details and franchise relationships
  - ``GET /health``: System health check & ChromaDB document counts
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Recommendation Schemas
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    """Input payload for ``POST /recommend``."""

    query: str = Field(
        ...,
        description="Natural language user recommendation prompt or taste query.",
        json_schema_extra={"example": "dark political fantasy with military strategy 2015+"},
    )
    year_min: Optional[int] = Field(
        None,
        description="Explicit year lower-bound constraint (overrides parser if provided).",
        json_schema_extra={"example": 2015},
    )
    year_max: Optional[int] = Field(
        None,
        description="Explicit year upper-bound constraint.",
    )
    format: Optional[str] = Field(
        None,
        description="Media format filter: TV, MOVIE, OVA, ONA, SPECIAL.",
        json_schema_extra={"example": "TV"},
    )
    min_score: Optional[int] = Field(
        None,
        description="Minimum average score filter (0-100).",
        json_schema_extra={"example": 75},
    )
    genres: list[str] = Field(
        default_factory=list,
        description="Explicit genre filters.",
        json_schema_extra={"example": ["Action", "Fantasy"]},
    )
    exclude_ids: list[int] = Field(
        default_factory=list,
        description="List of AniList IDs to exclude (e.g. already watched).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final recommendations to generate.",
    )


class RecommendedItem(BaseModel):
    """Single grounded recommendation item."""

    anilist_id: int = Field(description="Unique AniList show ID")
    title: str = Field(description="Title (Romaji or English)")
    year: Optional[int] = Field(None, description="Release year")
    format: Optional[str] = Field(None, description="Media format (TV, MOVIE, etc.)")
    score: Optional[int] = Field(None, description="Average AniList score")
    why: str = Field(description="2-3 sentence grounded justification citing tags and premise")
    matched_attributes: list[str] = Field(
        default_factory=list,
        description="Specific community tags or genres matched",
    )
    cover_image_url: Optional[str] = Field(None, description="URL to show cover image")
    site_url: Optional[str] = Field(None, description="AniList show web URL")


class CandidateProvenance(BaseModel):
    """Audit provenance item capturing retrieval and reranking scores."""

    anilist_id: int
    title: str
    similarity_score: float
    rerank_score: Optional[float] = None
    genres: str
    top_tags: str


class RecommendResponse(BaseModel):
    """Full recommendation response payload."""

    request_id: str
    query: str
    recommendations: list[RecommendedItem]
    reasoning_summary: str
    retrieved_provenance: list[CandidateProvenance] = Field(default_factory=list)
    relaxations_applied: list[str] = Field(default_factory=list)
    latency_ms: float


# ---------------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Single turn in a conversational chat session."""

    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str = Field(..., description="Message text")


class ChatFilterParams(BaseModel):
    """Optional active sidebar filters to ground conversational recommendations."""

    year_min: Optional[int] = Field(None, description="Year lower-bound filter.")
    year_max: Optional[int] = Field(None, description="Year upper-bound filter.")
    format: Optional[str] = Field(None, description="Media format filter.")
    min_score: Optional[int] = Field(None, description="Minimum average score filter.")
    genres: list[str] = Field(default_factory=list, description="Genre filters.")
    top_k: Optional[int] = Field(5, description="Number of items to retrieve.")
    exclude_ids: list[int] = Field(default_factory=list, description="Previously shown show IDs to exclude.")


class ChatRequest(BaseModel):
    """Input payload for ``POST /chat`` SSE streaming endpoint."""

    message: str = Field(..., description="Latest user message")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation",
    )
    active_filters: Optional[ChatFilterParams] = Field(
        None,
        description="Optional active sidebar filters to ground conversational recommendations",
    )


# ---------------------------------------------------------------------------
# Metadata & Health Schemas
# ---------------------------------------------------------------------------


class AnimeDetailResponse(BaseModel):
    """Detail response for ``GET /anime/{id}``."""

    anime_id: int
    id_mal: Optional[int] = None
    franchise_root_id: int
    title_romaji: str
    title_english: Optional[str] = None
    season_year: Optional[int] = None
    season: Optional[str] = None
    format: str
    status: str
    episodes: Optional[int] = None
    duration_mins: Optional[int] = None
    average_score: Optional[int] = None
    popularity: Optional[int] = None
    genres: list[str] = Field(default_factory=list)
    top_tags: list[str] = Field(default_factory=list)
    description_clean: str
    cover_image_url: Optional[str] = None
    site_url: Optional[str] = None


class HealthResponse(BaseModel):
    """System health check response for ``GET /health``."""

    status: str
    indexed_documents: int
    environment: str
    llm_provider: str
    version: str = "1.0.0"
