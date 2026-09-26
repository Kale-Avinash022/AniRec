"""User Feedback Thumbs-Up/Down Logger Component (Step 48).

Captures explicit user sentiment on generated recommendation sets and records
provenance traces to ``logs/feedback.jsonl`` for preference tuning and metrics.

Usage::

    from src.ui.feedback import render_feedback_buttons

    render_feedback_buttons(st, turn_id="turn_3", query="cyberpunk", response_text="...")
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logger import logger


def log_user_feedback(
    turn_id: str,
    query: str,
    response_text: str,
    rating: str,
    metadata: dict[str, Any] | None = None,
    feedback_file: Path | None = None,
) -> dict[str, Any]:
    """Persist a single user feedback event to the JSONL log file.

    Args:
        turn_id:       Unique identifier for the chat message turn.
        query:         The user query prompt that initiated the recommendation.
        response_text: The generated assistant response.
        rating:        "positive" | "negative".
        metadata:      Optional candidate list or latency metrics.
        feedback_file: Optional file path override (default: logs/feedback.jsonl).

    Returns:
        Structured record dictionary written to disk.
    """
    target_path = feedback_file or (settings.LOGS_DIR / "feedback.jsonl")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "turn_id": turn_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "response": response_text,
        "rating": rating,
        "metadata": metadata or {},
    }

    try:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Recorded user feedback [{turn_id}]: {rating}")
    except Exception as exc:
        logger.error(f"Failed to record feedback to {target_path}: {exc}")

    return record


def render_feedback_buttons(
    st_module: Any,
    turn_id: str,
    query: str,
    response_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Render interactive thumbs-up / thumbs-down buttons for a chat message."""
    col1, col2, _ = st_module.columns([1, 1, 8])

    existing_rating = st_module.session_state.feedback_history.get(turn_id)

    with col1:
        btn_label = "👍 Helpful" if existing_rating != "positive" else "✅ Liked"
        if st_module.button(btn_label, key=f"up_{turn_id}", disabled=(existing_rating is not None)):
            st_module.session_state.feedback_history[turn_id] = "positive"
            log_user_feedback(turn_id, query, response_text, "positive", metadata)
            st_module.toast("Thank you! Feedback recorded 👍")
            st_module.rerun()

    with col2:
        btn_label = "👎 Poor" if existing_rating != "negative" else "❌ Disliked"
        if st_module.button(btn_label, key=f"down_{turn_id}", disabled=(existing_rating is not None)):
            st_module.session_state.feedback_history[turn_id] = "negative"
            log_user_feedback(turn_id, query, response_text, "negative", metadata)
            st_module.toast("Thanks! We will refine future recommendations 👎")
            st_module.rerun()
