"""Tests for Semantic Vector Search with Metadata WHERE Clauses (Step 27)."""

import pytest
from unittest.mock import MagicMock
from src.retrieval.query_parser import ExtractedFilters
from src.retrieval.search import build_chroma_filter, search_candidates


def test_build_chroma_filter_empty():
    filters = ExtractedFilters()
    assert build_chroma_filter(filters) is None


def test_build_chroma_filter_single():
    filters = ExtractedFilters(year_min=2020)
    expected = {"season_year": {"$gte": 2020}}
    assert build_chroma_filter(filters) == expected


def test_build_chroma_filter_compound():
    filters = ExtractedFilters(
        year_min=2015,
        year_max=2022,
        format="TV",
        min_score=80,
        genres=["Action"],
    )
    result = build_chroma_filter(filters)
    assert "$and" in result
    conditions = result["$and"]
    assert {"season_year": {"$gte": 2015}} in conditions
    assert {"season_year": {"$lte": 2022}} in conditions
    assert {"format": {"$eq": "TV"}} in conditions
    assert {"average_score": {"$gte": 80}} in conditions
    assert {"genres": {"$contains": "Action"}} in conditions


def test_search_candidates_mocked():
    mock_store = MagicMock()
    mock_store.query.return_value = {
        "ids": [["1", "2"]],
        "documents": [["Doc 1", "Doc 2"]],
        "metadatas": [[{"title_romaji": "Anime 1"}, {"title_romaji": "Anime 2"}]],
        "distances": [[0.2, 0.4]],
    }

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    filters = ExtractedFilters(format="TV", min_score=75)
    candidates = search_candidates(
        store=mock_store,
        embedder=mock_embedder,
        semantic_text="psychological thriller",
        filters=filters,
        top_k=2,
    )

    assert len(candidates) == 2
    assert candidates[0]["id"] == "1"
    assert candidates[0]["similarity"] == 0.8
    assert candidates[1]["id"] == "2"
    assert candidates[1]["similarity"] == 0.6
    mock_store.query.assert_called_once()
