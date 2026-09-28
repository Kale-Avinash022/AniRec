"""Unit and integration tests for Query Intent Classification & Comparison RAG Pipeline."""

import json
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.generation.llm_client import MockChatModel
from src.generation.prompts import SYSTEM_COMPARISON_PROMPT
from src.retrieval.query_parser import extract_comparison_intent, parse_query_rules


@pytest.mark.parametrize(
    "query,expected_is_comp,expected_titles",
    [
        ("Eden Zero vs Fairy Tail", True, ["Eden Zero", "Fairy Tail"]),
        ("edens zero vs fairy tail", True, ["edens zero", "fairy tail"]),
        ("Death Note versus Code Geass", True, ["Death Note", "Code Geass"]),
        ("compare Steins;Gate and Erased", True, ["Steins;Gate", "Erased"]),
        ("difference between Naruto and Bleach", True, ["Naruto", "Bleach"]),
        ("Attack on Titan or Vinland Saga, which is better?", True, ["Attack on Titan", "Vinland Saga"]),
        ("is Jujutsu Kaisen better than Demon Slayer", True, ["Jujutsu Kaisen", "Demon Slayer"]),
        ("top 10 action anime", False, []),
        ("wholesome romance anime after 2020", False, []),
    ],
)
def test_extract_comparison_intent(query, expected_is_comp, expected_titles):
    is_comp, titles = extract_comparison_intent(query)
    assert is_comp == expected_is_comp
    assert titles == expected_titles


def test_parse_query_rules_comparison():
    parsed = parse_query_rules("Eden Zero vs Fairy Tail")
    assert parsed.is_comparison is True
    assert parsed.intent == "comparison"
    assert parsed.comparison_titles == ["Eden Zero", "Fairy Tail"]


@pytest.mark.asyncio
async def test_mock_chat_model_comparison_synthesis():
    llm = MockChatModel()
    candidates = [
        {
            "anilist_id": 1,
            "title": "Edens Zero",
            "year": 2021,
            "score": 75,
            "genres": "Action, Adventure, Sci-Fi",
            "tags": "space, magic, robots, friendship",
            "synopsis": "Shiki Granbell embarks on an interstellar quest aboard the Edens Zero.",
        },
        {
            "anilist_id": 2,
            "title": "Fairy Tail",
            "year": 2009,
            "score": 78,
            "genres": "Action, Adventure, Fantasy",
            "tags": "magic, wizards, guilds, dragons",
            "synopsis": "Lucy Heartfilia joins the boisterous Fairy Tail guild alongside Natsu Dragneel.",
        },
    ]
    prompt = (
        f"Head-to-Head Comparison Candidates:\n{json.dumps(candidates)}\n\n"
        f"User Comparison Request: Edens Zero vs Fairy Tail"
    )
    result = await llm.generate(prompt=prompt, system_prompt=SYSTEM_COMPARISON_PROMPT, json_mode=False)

    assert "Head-to-Head Comparison: Edens Zero vs Fairy Tail" in result
    assert "Premise & Worldbuilding" in result
    assert "Tone, Themes & Atmosphere" in result
    assert "Target Audience & Verdict" in result


@pytest.mark.asyncio
async def test_chat_stream_comparison_sse():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={
                "message": "Eden Zero vs Fairy Tail",
                "history": [],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        assert "event: context" in body
        assert "event: token" in body
        assert "event: done" in body
