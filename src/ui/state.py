"""Streamlit Session State Architecture & Initialization (Step 44).

Manages reactive state across browser interactions:
  - ``messages``: Multi-turn chat conversation history.
  - ``provenance_map``: Turn ID -> List of retrieved candidate shows & similarity scores.
  - ``feedback_history``: Turn ID -> User feedback ("up" | "down").
  - ``active_filters``: Active sidebar constraints (genres, year, score, format).
"""

from __future__ import annotations

from typing import Any

WELCOME_MESSAGE = (
    "👋 **Welcome to AniRec!** I'm your AI anime recommendation consultant.\n\n"
    "Tell me what kind of anime you're in the mood for — for example:\n"
    "- *'Dark political fantasy with military strategy 2015+'*\n"
    "- *'Wholesome comedy slice of life set in high school'* \n"
    "- *'Cyberpunk psychological thriller with high rating'*\n\n"
    "You can also use the sidebar on the left to set strict filters!"
)


def init_session_state(st_module: Any) -> None:
    """Initialize default session state keys if not already present."""
    if "messages" not in st_module.session_state:
        st_module.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    if "provenance_map" not in st_module.session_state:
        st_module.session_state.provenance_map = {}

    if "feedback_history" not in st_module.session_state:
        st_module.session_state.feedback_history = {}

    if "active_filters" not in st_module.session_state:
        st_module.session_state.active_filters = {}

    if "recommended_ids" not in st_module.session_state:
        st_module.session_state.recommended_ids = set()


def reset_session(st_module: Any) -> None:
    """Clear conversation history and restore default state."""
    st_module.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}
    ]
    st_module.session_state.provenance_map = {}
    st_module.session_state.feedback_history = {}
    st_module.session_state.recommended_ids = set()
