"""Tests for Interactive Sidebar Filters Component (Step 46)."""

import pytest
from unittest.mock import MagicMock
from src.ui.sidebar import render_sidebar_filters


class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class MockSidebar:
    def __init__(self):
        self.multiselect = MagicMock(return_value=["Action", "Fantasy"])
        self.slider = MagicMock(side_effect=[(2010, 2022), 80, 5])
        self.selectbox = MagicMock(return_value="TV")
        self.title = MagicMock()
        self.subheader = MagicMock()
        self.divider = MagicMock()
        self.button = MagicMock(return_value=False)


class MockStreamlitUI:
    def __init__(self):
        self.sidebar = MockSidebar()
        self.session_state = MockSessionState()
        self.rerun = MagicMock()


def test_render_sidebar_filters():
    mock_st = MockStreamlitUI()
    filters = render_sidebar_filters(mock_st)

    assert filters["genres"] == ["Action", "Fantasy"]
    assert filters["year_min"] == 2010
    assert filters["year_max"] == 2022
    assert filters["min_score"] == 80
    assert filters["format"] == "TV"
    assert filters["top_k"] == 5

    # Check session state was updated
    assert mock_st.session_state["active_filters"] == filters
