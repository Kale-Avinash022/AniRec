"""Tests for Streamlit Session State Initialization & Management (Step 44)."""

import pytest
from src.ui.state import WELCOME_MESSAGE, init_session_state, reset_session


class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()


def test_init_session_state():
    mock_st = MockStreamlit()
    init_session_state(mock_st)

    assert "messages" in mock_st.session_state
    assert len(mock_st.session_state.messages) == 1
    assert mock_st.session_state.messages[0]["content"] == WELCOME_MESSAGE
    assert "provenance_map" in mock_st.session_state
    assert "feedback_history" in mock_st.session_state
    assert "active_filters" in mock_st.session_state


def test_reset_session():
    mock_st = MockStreamlit()
    init_session_state(mock_st)

    # Simulate added turns
    mock_st.session_state.messages.append({"role": "user", "content": "hello"})
    mock_st.session_state.provenance_map[1] = [{"id": 1, "title": "Test"}]
    mock_st.session_state.feedback_history[1] = "up"

    assert len(mock_st.session_state.messages) == 2

    reset_session(mock_st)
    assert len(mock_st.session_state.messages) == 1
    assert mock_st.session_state.provenance_map == {}
    assert mock_st.session_state.feedback_history == {}
