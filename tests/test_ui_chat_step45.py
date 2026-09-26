"""Tests for Streamlit SSE Streaming Chat Consumer (Step 45)."""

import pytest
import json
from unittest.mock import MagicMock, patch
from src.ui.chat import stream_chat_response


def test_stream_chat_response_parsing():
    sse_lines = [
        "event: context",
        'data: [{"id": 16498, "title": "Attack on Titan"}]',
        "",
        "event: token",
        "data: I recommend ",
        "",
        "event: token",
        "data: Attack on Titan.",
        "",
        "event: done",
        "data: [DONE]",
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = sse_lines
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("httpx.stream", return_value=mock_resp):
        chunks = list(
            stream_chat_response(
                api_base_url="http://localhost:8000",
                message="Tell me an anime",
            )
        )

        tokens = [c[0] for c in chunks]
        candidates = chunks[0][1]

        assert "".join(tokens) == "I recommend Attack on Titan."
        assert len(candidates) == 1
        assert candidates[0]["id"] == 16498


def test_stream_chat_response_connection_error():
    with patch("httpx.stream", side_effect=Exception("Connection refused")):
        chunks = list(
            stream_chat_response(
                api_base_url="http://localhost:8000",
                message="Hello",
            )
        )
        assert len(chunks) == 1
        assert "Could not connect" in chunks[0][0]
