"""Tests for Progressive Filter Relaxation Strategy (Step 28)."""

import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.query_parser import ExtractedFilters
from src.retrieval.relaxation import search_with_relaxation


def test_strict_search_succeeds_no_relaxation():
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    dummy_candidates = [{"id": f"{i}", "similarity": 0.8} for i in range(10)]

    with patch("src.retrieval.relaxation.search_candidates", return_value=dummy_candidates) as mock_search:
        initial_filters = ExtractedFilters(format="TV", min_score=80)
        candidates, notes = search_with_relaxation(
            store=mock_store,
            embedder=mock_embedder,
            semantic_text="psychological thriller",
            initial_filters=initial_filters,
            min_results_needed=5,
        )

        assert len(candidates) == 10
        assert notes == []
        assert mock_search.call_count == 1


def test_progressive_relaxation_drops_score_first():
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    # Call 1 (strict): 1 result (< 5)
    # Call 2 (score dropped): 6 results (>= 5)
    call_1 = [{"id": "1", "similarity": 0.9}]
    call_2 = [{"id": f"{i}", "similarity": 0.8} for i in range(6)]

    with patch("src.retrieval.relaxation.search_candidates", side_effect=[call_1, call_2]) as mock_search:
        initial_filters = ExtractedFilters(format="TV", min_score=95)
        candidates, notes = search_with_relaxation(
            store=mock_store,
            embedder=mock_embedder,
            semantic_text="cyberpunk",
            initial_filters=initial_filters,
            min_results_needed=5,
        )

        assert len(candidates) == 6
        assert len(notes) == 1
        assert "Score filter" in notes[0]
        assert mock_search.call_count == 2


def test_progressive_relaxation_drops_year_when_needed():
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    # Call 1 (strict): 0
    # Call 2 (year dropped): 8
    call_1 = []
    call_2 = [{"id": f"{i}", "similarity": 0.75} for i in range(8)]

    with patch("src.retrieval.relaxation.search_candidates", side_effect=[call_1, call_2]) as mock_search:
        initial_filters = ExtractedFilters(year_min=2030)
        candidates, notes = search_with_relaxation(
            store=mock_store,
            embedder=mock_embedder,
            semantic_text="space mecha",
            initial_filters=initial_filters,
            min_results_needed=5,
        )

        assert len(candidates) == 8
        assert len(notes) == 1
        assert "Year filter" in notes[0]
