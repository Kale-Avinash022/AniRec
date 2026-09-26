"""Tests for User Feedback Logger Component (Step 48)."""

import pytest
import json
from unittest.mock import MagicMock
from pathlib import Path
from src.ui.feedback import log_user_feedback, render_feedback_buttons


class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class MockStreamlitFeedback:
    def __init__(self, click_up: bool = True, click_down: bool = False):
        self.columns = MagicMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        
        def mock_button(label, key=None, disabled=False):
            if key and key.startswith("up_"):
                return click_up
            if key and key.startswith("down_"):
                return click_down
            return False

        self.button = MagicMock(side_effect=mock_button)
        self.toast = MagicMock()
        self.rerun = MagicMock()
        self.session_state = MockSessionState({"feedback_history": {}})



def test_log_user_feedback(tmp_path: Path):
    feedback_file = tmp_path / "test_feedback.jsonl"

    record = log_user_feedback(
        turn_id="turn_1",
        query="cyberpunk",
        response_text="Recommended Akira.",
        rating="positive",
        metadata={"top_id": 16498},
        feedback_file=feedback_file,
    )

    assert record["turn_id"] == "turn_1"
    assert record["rating"] == "positive"
    assert feedback_file.exists()

    with open(feedback_file, "r", encoding="utf-8") as f:
        line = f.readline()
        saved = json.loads(line)
        assert saved["query"] == "cyberpunk"
        assert saved["rating"] == "positive"


def test_render_feedback_buttons():
    mock_st = MockStreamlitFeedback(click_up=True, click_down=False)

    render_feedback_buttons(
        st_module=mock_st,
        turn_id="turn_1",
        query="mecha",
        response_text="Recommended Gundam.",
    )

    mock_st.columns.assert_called_once_with([1, 1, 8])
    assert mock_st.session_state.feedback_history["turn_1"] == "positive"
    mock_st.toast.assert_called_once()
    mock_st.rerun.assert_called_once()

