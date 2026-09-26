"""Interactive Sidebar Filters & Session Controls (Step 46).

Renders the sidebar control panel allowing users to:
  1. Manually configure hard search constraints (Genres, Year Range, Min Score, Format).
  2. Adjust recommendation count (top_k).
  3. Reset the chat conversation history.
  4. View active vector catalog index health stats.

Usage::

    from src.ui.sidebar import render_sidebar_filters

    active_filters = render_sidebar_filters(st)
"""

from __future__ import annotations

from typing import Any

from src.retrieval.query_parser import KNOWN_GENRES
from src.ui.state import reset_session

AVAILABLE_FORMATS: list[str] = ["TV", "MOVIE", "OVA", "ONA", "SPECIAL"]


def render_sidebar_filters(st_module: Any) -> dict[str, Any]:
    """Render interactive sidebar widgets and return structured filter dictionary."""
    st_module.sidebar.title("⚡ Search Controls")

    # ── Genre Filter ──────────────────────────────────────────────────────────
    st_module.sidebar.subheader("🎨 Genres")
    selected_genres = st_module.sidebar.multiselect(
        "Filter by Genre(s)",
        options=KNOWN_GENRES,
        default=[],
        help="Select one or more genres to strictly require.",
    )

    # ── Release Year Range Slider ─────────────────────────────────────────────
    st_module.sidebar.subheader("📅 Release Year")
    year_range = st_module.sidebar.slider(
        "Year Range",
        min_value=1970,
        max_value=2026,
        value=(1980, 2026),
        help="Filter shows released within this year window.",
    )

    # ── Score Slider ──────────────────────────────────────────────────────────
    st_module.sidebar.subheader("⭐ Minimum Score")
    min_score = st_module.sidebar.slider(
        "AniList Rating (0 - 100)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        help="Require shows with an average score at or above this threshold.",
    )

    # ── Format Selector ───────────────────────────────────────────────────────
    st_module.sidebar.subheader("🎬 Media Format")
    selected_format = st_module.sidebar.selectbox(
        "Format",
        options=["Any"] + AVAILABLE_FORMATS,
        index=0,
        help="Restrict recommendations to specific release formats.",
    )

    # ── Results Count ─────────────────────────────────────────────────────────
    top_k = st_module.sidebar.slider(
        "Recommendations Count",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of recommendations to generate (1 - 20).",
    )

    # ── Reset Session Button ──────────────────────────────────────────────────
    st_module.sidebar.divider()
    if st_module.sidebar.button("🔄 Clear Chat History", use_container_width=True):
        reset_session(st_module)
        st_module.rerun()

    format_val = selected_format if selected_format != "Any" else None
    year_min = year_range[0] if year_range[0] > 1970 else None
    year_max = year_range[1] if year_range[1] < 2026 else None
    score_val = min_score if min_score > 0 else None

    filters_dict = {
        "genres": selected_genres,
        "year_min": year_min,
        "year_max": year_max,
        "min_score": score_val,
        "format": format_val,
        "top_k": top_k,
    }

    # Store in session state for cross-component access
    st_module.session_state.active_filters = filters_dict

    return filters_dict
