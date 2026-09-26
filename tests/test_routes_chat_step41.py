"""Tests for POST /chat Multi-Turn SSE Streaming Endpoint (Step 41)."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.generation.llm_client import MockChatModel


@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def test_chat_stream_endpoint_successful(test_client):
    mock_candidates = [
        {
            "id": "16498",
            "similarity": 0.65,
            "document": "Attack on Titan description",
            "metadata": {
                "anilist_id": 16498,
                "title_romaji": "Shingeki no Kyojin",
                "season_year": 2013,
                "format": "TV",
                "average_score": 85,
                "genres": "Action",
                "top_tags": "military (88)",
            },
        }
    ]

    mock_llm = MockChatModel(response_text="I recommend Attack on Titan for dark military action.")

    with patch("src.api.routes_chat.search_with_relaxation", return_value=(mock_candidates, [])), \
         patch("src.api.routes_chat.get_services") as mock_services:

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = mock_candidates

        mock_recommender = MagicMock()
        mock_recommender.llm = mock_llm
        mock_services.return_value = (MagicMock(), MagicMock(), mock_reranker, mock_recommender)

        payload = {
            "message": "Give me something military with dark fantasy",
            "history": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello! What kind of anime do you enjoy?"},
            ],
        }

        response = test_client.post("/chat", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        body = response.text
        assert "event: context" in body
        assert "event: token" in body
        assert "event: done" in body
        assert "[DONE]" in body
        assert "Shingeki no Kyojin" in body


def test_chat_stream_empty_message_400(test_client):
    response = test_client.post("/chat", json={"message": "   "})
    assert response.status_code == 400
