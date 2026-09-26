"""Tests for UI Components & Explainability Panels (Step 47)."""

import pytest
from unittest.mock import MagicMock
from src.ui.components import render_provenance_expander, render_recommendation_card


class MockExpanderContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockStreamlitComponents:
    def __init__(self):
        self.expander = MagicMock(return_value=MockExpanderContext())
        self.markdown = MagicMock()
        self.caption = MagicMock()
        self.info = MagicMock()
        self.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        self.image = MagicMock()
        self.write = MagicMock()
        self.divider = MagicMock()


def test_render_provenance_expander():
    mock_st = MockStreamlitComponents()
    candidates = [
        {
            "anilist_id": 16498,
            "title": "Attack on Titan",
            "similarity_score": 0.65,
            "rerank_score": 4.5,
            "matched_tags": "military (88), dark fantasy (77)",
        }
    ]

    render_provenance_expander(mock_st, candidates, relaxations=["Score relaxed"])
    mock_st.expander.assert_called_once()
    mock_st.info.assert_called_once()
    assert any("Attack on Titan" in str(call) for call in mock_st.markdown.call_args_list)


def test_render_recommendation_card():
    mock_st = MockStreamlitComponents()
    show = {
        "anilist_id": 16498,
        "title": "Attack on Titan",
        "year": 2013,
        "format": "TV",
        "score": 85,
        "why": "High-stakes military strategy.",
        "matched_attributes": ["military (88)", "dark fantasy (77)"],
        "cover_image_url": "https://example.com/aot.jpg",
    }

    render_recommendation_card(mock_st, show)
    mock_st.columns.assert_called_once_with([1, 4])
    mock_st.divider.assert_called_once()
