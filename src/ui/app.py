"""AniRec Streamlit Application (Phases 8 Entrypoint).

Production conversational anime recommendation and discovery interface.

Run with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from src.core.config import settings
from src.ui.chat import stream_chat_response
from src.ui.components import (
    parse_streamed_suggestions,
    render_aligned_recommendation_card,
    render_comparison_view,
)
from src.ui.feedback import render_feedback_buttons
from src.ui.sidebar import render_sidebar_filters
from src.ui.state import init_session_state


# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AniRec | Anime Discovery RAG Engine",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #FF4B4B;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 0.5rem;
        border-radius: 8px;
        background: #f8f9fa;
        margin-bottom: 0.5rem;
    }
    /* Anime poster styling */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        object-fit: cover;
        width: 100%;
        max-height: 280px;
    }
    div[data-testid="stImage"] img:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Initialize State ───────────────────────────────────────────────────────
init_session_state(st)

# ── Render Sidebar ─────────────────────────────────────────────────────────
active_filters = render_sidebar_filters(st)

# ── Main Header ────────────────────────────────────────────────────
st.markdown("<div class='main-header'>🌸 AniRec — Conversational Anime Discovery</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Powered by Hybrid Vector Retrieval, Cross-Encoder Reranking & Grounded LLM Generation</div>",
    unsafe_allow_html=True,
)

# API Base URL configuration
API_BASE_URL = os.environ.get("API_BASE_URL", f"http://localhost:{settings.PORT}")


def _render_assistant_message(msg_content: str, candidates: list[dict]) -> None:
    """Render an assistant response:
    - If head-to-head comparison: side-by-side comparison view
    - If recommendation list (1. Title - reason): aligned recommendation cards with posters
    - Otherwise: standard markdown
    """
    if not candidates:
        st.markdown(msg_content)
        return

    # Check for head-to-head comparison
    msg_lower = msg_content.lower()
    is_comparison = (
        len(candidates) == 2
        and ("head-to-head" in msg_lower or " vs " in msg_lower or "comparison:" in msg_lower or "⚔️" in msg_content)
    )
    if is_comparison:
        render_comparison_view(st, candidates, msg_content)
        return

    intro_text, descriptions_by_rank = parse_streamed_suggestions(msg_content)
    if not descriptions_by_rank:
        st.markdown(msg_content)
        return

    if intro_text:
        st.markdown(intro_text)

    for rank, candidate in enumerate(candidates, 1):
        desc = descriptions_by_rank.get(rank) or candidate.get("why") or candidate.get("synopsis")
        render_aligned_recommendation_card(
            st_module=st,
            rank=rank,
            show=candidate,
            description_override=desc,
        )



# ── Render Conversation History ────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🌸" if msg["role"] == "assistant" else "👤"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            turn_key = f"turn_{idx}"
            candidates = st.session_state.provenance_map.get(turn_key, [])
            _render_assistant_message(msg["content"], candidates)

            if idx > 0:
                user_query = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
                render_feedback_buttons(st, turn_id=turn_key, query=user_query, response_text=msg["content"])

# ── Chat Input & Streaming Execution ───────────────────────────────────────
user_input = st.chat_input("Ask for anime recommendations (e.g. '90s cyberpunk thriller' or 'wholesome romance')...")

if user_input:
    # 1. Append user message to state & render
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 2. Stream assistant response
    with st.chat_message("assistant", avatar="🌸"):
        response_placeholder = st.empty()
        full_response = ""
        captured_candidates: list[dict] = []
        # Attach previously shown IDs to active filters to prevent repetition
        filters_with_exclusions = dict(active_filters or {})
        filters_with_exclusions["exclude_ids"] = list(st.session_state.recommended_ids)

        with st.spinner("Searching catalog and generating grounded recommendations..."):
            for token_chunk, candidates in stream_chat_response(
                api_base_url=API_BASE_URL,
                message=user_input,
                history=st.session_state.messages[:-1],
                active_filters=filters_with_exclusions,
            ):
                full_response += token_chunk
                response_placeholder.markdown(full_response + "▌")
                if candidates and not captured_candidates:
                    captured_candidates = candidates

        response_placeholder.empty()
        _render_assistant_message(full_response, captured_candidates)

    # 3. Store assistant message and provenance in state
    turn_idx = len(st.session_state.messages)
    turn_key = f"turn_{turn_idx}"
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    if captured_candidates:
        st.session_state.provenance_map[turn_key] = captured_candidates
        # Track recommended IDs to avoid repeating the same shows in subsequent turns
        for c in captured_candidates:
            aid = c.get("anilist_id") or c.get("id")
            if aid:
                try:
                    int_id = int(str(aid).replace("anilist_", "").strip())
                    st.session_state.recommended_ids.add(int_id)
                except (ValueError, TypeError):
                    pass

    st.rerun()
