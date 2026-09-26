"""Tests for Strict JSON Mode Output Contract (Step 34)."""

import pytest
from src.generation.contract import (
    RecommendedShow,
    RecommendationOutput,
    extract_json_payload,
    parse_llm_json,
)


def test_schema_instantiation():
    show = RecommendedShow(
        anilist_id=16498,
        title="Attack on Titan",
        year=2013,
        why="High stakes grimdark military action.",
        matched_attributes=["military (88)", "dark fantasy (77)"],
    )
    output = RecommendationOutput(
        recommendations=[show],
        reasoning_summary="Selected dark action shows.",
    )
    assert len(output.recommendations) == 1
    assert output.recommendations[0].anilist_id == 16498


def test_extract_json_payload_markdown_fences():
    fenced = """```json
{
  "recommendations": [
    {
      "anilist_id": 9253,
      "title": "Steins;Gate",
      "year": 2011,
      "why": "Sci-Fi thriller about time travel.",
      "matched_attributes": ["time travel", "psychological"]
    }
  ],
  "reasoning_summary": "Top sci-fi pick."
}
```"""
    parsed = parse_llm_json(fenced)
    assert len(parsed.recommendations) == 1
    assert parsed.recommendations[0].anilist_id == 9253
    assert parsed.recommendations[0].title == "Steins;Gate"


def test_parse_llm_json_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        parse_llm_json("This is definitely not JSON.")


def test_parse_llm_json_schema_violation_raises():
    # Missing required 'why' and 'title' fields
    bad_json = '{"recommendations": [{"anilist_id": "not_an_int"}]}'
    with pytest.raises(ValueError, match="LLM output violated contract schema"):
        parse_llm_json(bad_json)
