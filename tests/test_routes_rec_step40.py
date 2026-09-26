"""Tests for POST /recommend One-Shot Endpoint (Step 40)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.generation.contract import RecommendationOutput, RecommendedShow


@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def test_recommend_endpoint_successful(test_client):
    mock_candidates = [
        {
            "id": "16498",
            "similarity": 0.65,
            "rerank_score": 4.5,
            "document": "In a world surrounded by giant walls...",
            "metadata": {
                "anilist_id": 16498,
                "title_romaji": "Shingeki no Kyojin",
                "title_english": "Attack on Titan",
                "season_year": 2013,
                "format": "TV",
                "average_score": 85,
                "genres": "Action, Drama, Fantasy",
                "top_tags": "military (88), dark fantasy (77)",
                "cover_image_url": "https://example.com/aot.jpg",
                "site_url": "https://anilist.co/anime/16498",
            },
        }
    ]

    mock_gen_output = RecommendationOutput(
        recommendations=[
            RecommendedShow(
                anilist_id=16498,
                title="Attack on Titan",
                year=2013,
                why="High stakes military strategy and grimdark setting.",
                matched_attributes=["military (88)", "dark fantasy (77)"],
            )
        ],
        reasoning_summary="Selected top military series.",
    )

    with patch("src.api.routes_rec.search_with_relaxation", return_value=(mock_candidates, [])) as mock_search, \
         patch("src.api.routes_rec.GroundedRecommender.recommend", new_callable=AsyncMock, return_value=mock_gen_output):

        payload = {
            "query": "dark military fantasy anime",
            "format": "TV",
            "min_score": 75,
            "top_k": 5,
        }

        response = test_client.post("/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert len(data["recommendations"]) == 1
        assert data["recommendations"][0]["anilist_id"] == 16498
        assert data["recommendations"][0]["title"] == "Attack on Titan"
        assert len(data["retrieved_provenance"]) == 1
        assert data["retrieved_provenance"][0]["similarity_score"] == 0.65
        assert data["latency_ms"] > 0


def test_recommend_endpoint_not_found(test_client):
    with patch("src.api.routes_rec.search_with_relaxation", return_value=([], ["All relaxed"])):
        payload = {"query": "some completely impossible anime query 9999"}
        response = test_client.post("/recommend", json=payload)
        assert response.status_code == 404
        assert "No anime candidates found" in response.json()["detail"]
