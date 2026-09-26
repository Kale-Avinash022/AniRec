"""AniRec Streamlit UI package.

Phase 8: Frontend Chat UI
  Step 44 — state.py      : Session state architecture & initialization
  Step 45 — chat.py       : SSE streaming chat consumer & markdown renderer
  Step 46 — sidebar.py    : Interactive sidebar filters
  Step 47 — components.py : "Why This Recommendation?" Explainability panel
  Step 48 — feedback.py   : User feedback thumbs up/down logger
  App Entrypoint: app.py
"""

from src.ui.chat import stream_chat_response
from src.ui.components import render_provenance_expander, render_recommendation_card
from src.ui.feedback import log_user_feedback, render_feedback_buttons
from src.ui.sidebar import render_sidebar_filters
from src.ui.state import WELCOME_MESSAGE, init_session_state, reset_session

__all__ = [
    "WELCOME_MESSAGE",
    "init_session_state",
    "reset_session",
    "stream_chat_response",
    "render_sidebar_filters",
    "render_provenance_expander",
    "render_recommendation_card",
    "log_user_feedback",
    "render_feedback_buttons",
]
