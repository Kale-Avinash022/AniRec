"""Tests for Anti-Hallucination Post-Generation Verification Gate (Step 35)."""

import pytest
from src.generation.contract import RecommendationOutput, RecommendedShow
from src.generation.verification import (
    build_candidate_whitelist,
    verify_recommendations_against_candidates,
)


def test_verification_all_valid():
    candidates = [
        {"id": "16498", "metadata": {"anilist_id": 16498, "title_romaji": "Shingeki no Kyojin"}},
        {"id": "9253", "metadata": {"anilist_id": 9253, "title_romaji": "Steins;Gate"}},
    ]

    output = RecommendationOutput(
        recommendations=[
            RecommendedShow(anilist_id=16498, title="Shingeki no Kyojin", why="Great action."),
            RecommendedShow(anilist_id=9253, title="Steins;Gate", why="Great sci-fi."),
        ],
        reasoning_summary="Valid recommendations.",
    )

    verified, is_faithful = verify_recommendations_against_candidates(output, candidates)
    assert is_faithful is True
    assert len(verified.recommendations) == 2


def test_verification_drops_hallucinated_shows():
    candidates = [
        {"id": "16498", "metadata": {"anilist_id": 16498, "title_romaji": "Shingeki no Kyojin"}},
    ]

    output = RecommendationOutput(
        recommendations=[
            RecommendedShow(anilist_id=16498, title="Shingeki no Kyojin", why="Real show in candidates."),
            RecommendedShow(anilist_id=999999, title="Made Up Anime 2099", why="Fictional hallucinated show."),
        ],
        reasoning_summary="Mixed recommendations.",
    )

    verified, is_faithful = verify_recommendations_against_candidates(output, candidates)
    assert is_faithful is False
    assert len(verified.recommendations) == 1
    assert verified.recommendations[0].anilist_id == 16498


def test_verification_corrects_mismatched_id_by_title():
    candidates = [
        {"id": "16498", "metadata": {"anilist_id": 16498, "title_romaji": "Shingeki no Kyojin", "title_english": "Attack on Titan"}},
    ]

    output = RecommendationOutput(
        recommendations=[
            # LLM output had title "Attack on Titan" but id was 0 or incorrect
            RecommendedShow(anilist_id=0, title="Attack on Titan", why="Matched title correctly."),
        ],
        reasoning_summary="Testing title matching.",
    )

    verified, is_faithful = verify_recommendations_against_candidates(output, candidates)
    assert is_faithful is True
    assert len(verified.recommendations) == 1
    assert verified.recommendations[0].anilist_id == 16498
