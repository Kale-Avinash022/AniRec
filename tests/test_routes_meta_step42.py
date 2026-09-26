"""Tests for GET /health and GET /anime/{id} Endpoints (Step 42)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def test_health_check_endpoint(test_client):
    mock_store = MagicMock()
    mock_store.count.return_value = 3000

    with patch("src.api.routes_meta.get_services") as mock_services:
        mock_services.return_value = (mock_store, MagicMock(), MagicMock(), MagicMock())

        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["indexed_documents"] == 3000
        assert "version" in data


def test_get_anime_detail_found(test_client):
    mock_store = MagicMock()
    mock_store.collection.get.return_value = {
        "ids": ["anilist_16498"],
        "documents": ["In a world with giant walls..."],
        "metadatas": [
            {
                "anime_id": 16498,
                "title_romaji": "Shingeki no Kyojin",
                "title_english": "Attack on Titan",
                "season_year": 2013,
                "format": "TV",
                "average_score": 85,
                "genres": "Action, Drama, Fantasy",
                "top_tags": "military (88), dark fantasy (77)",
            }
        ],
    }

    with patch("src.api.routes_meta.get_services") as mock_services:
        mock_services.return_value = (mock_store, MagicMock(), MagicMock(), MagicMock())

        response = test_client.get("/anime/16498")
        assert response.status_code == 200
        data = response.json()
        assert data["anime_id"] == 16498
        assert data["title_romaji"] == "Shingeki no Kyojin"
        assert data["title_english"] == "Attack on Titan"
        assert "Action" in data["genres"]


def test_get_anime_detail_not_found(test_client):
    mock_store = MagicMock()
    mock_store.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}

    with patch("src.api.routes_meta.get_services") as mock_services:
        mock_services.return_value = (mock_store, MagicMock(), MagicMock(), MagicMock())

        response = test_client.get("/anime/9999999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
