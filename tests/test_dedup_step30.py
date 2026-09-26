"""Tests for Franchise Deduplication & Lineage Collapse Algorithm (Step 30)."""

import pytest
from src.retrieval.dedup import deduplicate_franchises


def test_deduplicate_empty():
    assert deduplicate_franchises([]) == []


def test_deduplicate_franchises_max_1():
    candidates = [
        {
            "id": "16498",
            "metadata": {"title_romaji": "Shingeki no Kyojin S1", "franchise_root_id": 16498, "anime_id": 16498},
            "rerank_score": 4.5,
        },
        {
            "id": "20958",
            "metadata": {"title_romaji": "Shingeki no Kyojin S2", "franchise_root_id": 16498, "anime_id": 20958},
            "rerank_score": 4.2,
        },
        {
            "id": "101347",
            "metadata": {"title_romaji": "Vinland Saga", "franchise_root_id": 101347, "anime_id": 101347},
            "rerank_score": 3.9,
        },
        {
            "id": "99147",
            "metadata": {"title_romaji": "Shingeki no Kyojin S3", "franchise_root_id": 16498, "anime_id": 99147},
            "rerank_score": 3.8,
        },
        {
            "id": "9253",
            "metadata": {"title_romaji": "Steins;Gate", "franchise_root_id": 9253, "anime_id": 9253},
            "rerank_score": 3.5,
        },
    ]

    deduped = deduplicate_franchises(candidates, max_per_franchise=1)

    assert len(deduped) == 3
    # First entry of AoT kept
    assert deduped[0]["metadata"]["title_romaji"] == "Shingeki no Kyojin S1"
    # Vinland Saga kept
    assert deduped[1]["metadata"]["title_romaji"] == "Vinland Saga"
    # Steins;Gate kept
    assert deduped[2]["metadata"]["title_romaji"] == "Steins;Gate"


def test_deduplicate_franchises_max_2():
    candidates = [
        {"id": "1", "metadata": {"title_romaji": "Show A Part 1", "franchise_root_id": 100}},
        {"id": "2", "metadata": {"title_romaji": "Show A Part 2", "franchise_root_id": 100}},
        {"id": "3", "metadata": {"title_romaji": "Show A Movie", "franchise_root_id": 100}},
        {"id": "4", "metadata": {"title_romaji": "Show B", "franchise_root_id": 200}},
    ]

    deduped = deduplicate_franchises(candidates, max_per_franchise=2)
    assert len(deduped) == 3
    assert [c["id"] for c in deduped] == ["1", "2", "4"]


def test_deduplicate_fallback_without_root_id():
    candidates = [
        {"id": "10", "metadata": {"title_romaji": "Standalone 1"}},
        {"id": "20", "metadata": {"title_romaji": "Standalone 2"}},
    ]
    deduped = deduplicate_franchises(candidates, max_per_franchise=1)
    assert len(deduped) == 2
