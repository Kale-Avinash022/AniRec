"""Tests for Pydantic Request/Response API Schemas (Step 39)."""

import pytest
from pydantic import ValidationError
from src.api.schemas import (
    RecommendRequest,
    RecommendedItem,
    CandidateProvenance,
    RecommendResponse,
    ChatMessage,
    ChatRequest,
    HealthResponse,
    AnimeDetailResponse,
)


def test_recommend_request_validation():
    req = RecommendRequest(
        query="cyberpunk anime 2020+",
        year_min=2020,
        format="TV",
        min_score=80,
        genres=["Sci-Fi", "Action"],
        exclude_ids=[16498],
        top_k=5,
    )
    assert req.query == "cyberpunk anime 2020+"
    assert req.year_min == 2020
    assert req.format == "TV"
    assert req.top_k == 5


def test_recommend_request_missing_query_raises():
    with pytest.raises(ValidationError):
        RecommendRequest()  # query is required


def test_recommend_response_structure():
    item = RecommendedItem(
        anilist_id=16498,
        title="Attack on Titan",
        year=2013,
        format="TV",
        score=85,
        why="Military dark fantasy with strategy.",
        matched_attributes=["military (88)", "dark fantasy (77)"],
    )
    prov = CandidateProvenance(
        anilist_id=16498,
        title="Shingeki no Kyojin",
        similarity_score=0.55,
        rerank_score=4.5,
        genres="Action, Drama, Fantasy",
        top_tags="military (88), dark fantasy (77)",
    )
    resp = RecommendResponse(
        request_id="req-12345",
        query="dark military anime",
        recommendations=[item],
        reasoning_summary="Top military picks.",
        retrieved_provenance=[prov],
        relaxations_applied=["Score relaxed"],
        latency_ms=125.5,
    )

    assert resp.request_id == "req-12345"
    assert len(resp.recommendations) == 1
    assert len(resp.retrieved_provenance) == 1
    assert resp.latency_ms == 125.5


def test_chat_schemas():
    msg = ChatMessage(role="user", content="Recommend something similar to Psycho-Pass")
    filters = {"year_min": 2010, "genres": ["Action"]}
    req = ChatRequest(message="Recommend something", history=[msg], active_filters=filters)
    assert req.message == "Recommend something"
    assert len(req.history) == 1
    assert req.active_filters.year_min == 2010
    assert req.active_filters.genres == ["Action"]


def test_health_response():
    health = HealthResponse(
        status="healthy",
        indexed_documents=3000,
        environment="development",
        llm_provider="openai",
    )
    assert health.status == "healthy"
    assert health.indexed_documents == 3000
