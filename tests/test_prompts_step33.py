"""Tests for Grounded System Prompt Engineering & Context Assembly (Step 33)."""

import pytest
import json
from src.generation.prompts import (
    SYSTEM_RECOMMENDER_PROMPT,
    SYSTEM_CHAT_PROMPT,
    format_candidate_context,
    format_user_prompt,
)


def test_system_prompt_contains_crucial_guardrails():
    assert "Candidate Grounding" in SYSTEM_RECOMMENDER_PROMPT or "Grounding" in SYSTEM_RECOMMENDER_PROMPT
    assert "Anti-Spoiler" in SYSTEM_RECOMMENDER_PROMPT
    assert "JSON" in SYSTEM_RECOMMENDER_PROMPT
    assert "recommendations" in SYSTEM_RECOMMENDER_PROMPT


def test_format_candidate_context():
    cand = {
        "id": "16498",
        "document": "In a world surrounded by enormous walls...",
        "metadata": {
            "anime_id": 16498,
            "title_romaji": "Shingeki no Kyojin",
            "title_english": "Attack on Titan",
            "season_year": 2013,
            "format": "TV",
            "average_score": 85,
            "genres": "Action, Drama, Fantasy",
            "top_tags": "military (88), dark fantasy (77)",
        },
    }

    ctx = format_candidate_context(cand)
    assert ctx["anilist_id"] == 16498
    assert ctx["title"] == "Attack on Titan"
    assert ctx["year"] == 2013
    assert ctx["format"] == "TV"
    assert ctx["score"] == 85
    assert "military" in ctx["tags"]
    assert "surrounded by enormous walls" in ctx["synopsis"]


def test_format_user_prompt():
    candidates = [
        {
            "id": "1",
            "document": "Doc 1",
            "metadata": {"title_romaji": "Anime 1", "season_year": 2020},
        },
        {
            "id": "2",
            "document": "Doc 2",
            "metadata": {"title_romaji": "Anime 2", "season_year": 2021},
        },
    ]

    prompt = format_user_prompt("Suggest me 90s cyberpunk", candidates)
    assert "User Request: Suggest me 90s cyberpunk" in prompt
    assert "Candidate Shows:" in prompt
    assert "Anime 1" in prompt
    assert "Anime 2" in prompt
