"""AniRec FastAPI backend module.

Phase 7: Backend API Engineering
  Step 38 — main.py      : FastAPI app initialization, middleware & CORS
  Step 39 — schemas.py   : Pydantic request/response schemas
  Step 40 — routes_rec.py: POST /recommend one-shot endpoint
  Step 41 — routes_chat.py: POST /chat SSE streaming endpoint
  Step 42 — routes_meta.py: GET /anime/{id} and GET /health endpoints
  Step 43 — telemetry.py : API latency & cost tracking middleware
"""

from src.api.main import app, create_app
from src.api.schemas import (
    AnimeDetailResponse,
    CandidateProvenance,
    ChatFilterParams,
    ChatMessage,
    ChatRequest,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RecommendedItem,
)

__all__ = [
    "app",
    "create_app",
    "RecommendRequest",
    "RecommendedItem",
    "CandidateProvenance",
    "RecommendResponse",
    "ChatMessage",
    "ChatFilterParams",
    "ChatRequest",
    "AnimeDetailResponse",
    "HealthResponse",
]
