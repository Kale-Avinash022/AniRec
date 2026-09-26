"""Tests for Cross-Encoder Reranker Integration (Step 29)."""

import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.reranker import Reranker


def test_reranker_empty_candidates():
    with patch("sentence_transformers.CrossEncoder"):
        reranker = Reranker(model_name="mock-model")
        assert reranker.rerank(query="any", candidates=[]) == []


def test_reranker_sorting_and_top_k():
    mock_cross_encoder_instance = MagicMock()
    # Predict returns scores [0.1, 0.9, 0.5] for 3 candidates
    mock_cross_encoder_instance.predict.return_value = [0.1, 0.9, 0.5]

    with patch("sentence_transformers.CrossEncoder", return_value=mock_cross_encoder_instance):
        reranker = Reranker(model_name="mock-model")

        candidates = [
            {"id": "doc_1", "document": "Doc 1 text", "metadata": {"title_romaji": "Anime 1"}},
            {"id": "doc_2", "document": "Doc 2 text", "metadata": {"title_romaji": "Anime 2"}},
            {"id": "doc_3", "document": "Doc 3 text", "metadata": {"title_romaji": "Anime 3"}},
        ]

        reranked = reranker.rerank(
            query="action anime",
            candidates=candidates,
            top_k=2,
        )

        assert len(reranked) == 2
        # Highest score (0.9) must be first: doc_2
        assert reranked[0]["id"] == "doc_2"
        assert reranked[0]["rerank_score"] == 0.9

        # Second highest score (0.5) must be second: doc_3
        assert reranked[1]["id"] == "doc_3"
        assert reranked[1]["rerank_score"] == 0.5
