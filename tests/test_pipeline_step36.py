"""Tests for Generation Self-Correction & Fallback Pipeline (Step 36)."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from src.generation.contract import RecommendationOutput, RecommendedShow
from src.generation.llm_client import MockChatModel
from src.generation.pipeline import GroundedRecommender, build_deterministic_fallback


@pytest.mark.asyncio
async def test_recommender_successful_flow():
    candidates = [
        {
            "id": "16498",
            "document": "Attack on Titan description",
            "metadata": {"anilist_id": 16498, "title_romaji": "Shingeki no Kyojin", "season_year": 2013},
        }
    ]

    mock_json = json.dumps({
        "recommendations": [
            {
                "anilist_id": 16498,
                "title": "Shingeki no Kyojin",
                "year": 2013,
                "why": "High-stakes military strategy and grimdark fantasy.",
                "matched_attributes": ["military (88)", "dark fantasy (77)"],
            }
        ],
        "reasoning_summary": "Top dark military anime.",
    })

    llm = MockChatModel(response_text=mock_json)
    recommender = GroundedRecommender(llm=llm)

    output = await recommender.recommend("dark military anime", candidates)
    assert len(output.recommendations) == 1
    assert output.recommendations[0].anilist_id == 16498


@pytest.mark.asyncio
async def test_recommender_triggers_self_correction_on_hallucination():
    candidates = [
        {
            "id": "16498",
            "document": "AoT description",
            "metadata": {"anilist_id": 16498, "title_romaji": "Shingeki no Kyojin"},
        }
    ]

    # Turn 1: Hallucinated show ID 999999
    turn1_json = json.dumps({
        "recommendations": [
            {"anilist_id": 999999, "title": "Fake Anime", "why": "Invented show."}
        ],
        "reasoning_summary": "Bad summary.",
    })

    # Turn 2: Corrected show ID 16498
    turn2_json = json.dumps({
        "recommendations": [
            {"anilist_id": 16498, "title": "Shingeki no Kyojin", "why": "Corrected show."}
        ],
        "reasoning_summary": "Good summary.",
    })

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=[turn1_json, turn2_json])

    recommender = GroundedRecommender(llm=mock_llm)
    output = await recommender.recommend("action", candidates, max_retries=1)

    assert mock_llm.generate.call_count == 2
    assert len(output.recommendations) == 1
    assert output.recommendations[0].anilist_id == 16498


@pytest.mark.asyncio
async def test_deterministic_fallback_when_llm_fails():
    candidates = [
        {
            "id": "9253",
            "document": "Steins;Gate synopsis",
            "metadata": {
                "anilist_id": 9253,
                "title_romaji": "Steins;Gate",
                "title_english": "Steins;Gate",
                "season_year": 2011,
                "genres": "Drama, Sci-Fi, Thriller",
                "top_tags": "time travel (92), sci-fi (88)",
            },
        }
    ]

    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(side_effect=RuntimeError("API Connection Failure"))

    recommender = GroundedRecommender(llm=mock_llm)
    output = await recommender.recommend("sci-fi thriller", candidates)

    assert len(output.recommendations) == 1
    assert output.recommendations[0].anilist_id == 9253
    assert output.recommendations[0].title == "Steins;Gate"
    assert "Fallback" in output.reasoning_summary or "Retrieved" in output.reasoning_summary
