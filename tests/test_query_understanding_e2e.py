"""Comprehensive End-to-End Tests for Query Understanding (Step 25 & Step 26)."""

import pytest
from src.retrieval.query_parser import parse_query_rules, ExtractedFilters, ParsedQuery
from src.retrieval.query_expander import expand_query
from src.retrieval.llm_query_parser import parse_query, parse_query_with_llm_sync
from unittest.mock import patch, MagicMock, AsyncMock
from src.retrieval.llm_query_parser import LLMExtractedQuery


# ===========================================================================
# Step 25: Rules-Based Query Filter Parser Tests
# ===========================================================================

@pytest.mark.parametrize(
    "query,expected_year_min,expected_year_max,expected_format,expected_min_score,expected_genres",
    [
        (
            "90s mecha anime",
            1990,
            1999,
            None,
            None,
            ["Mecha"],
        ),
        (
            "wholesome romance anime TV series 2018 or later score >= 80",
            2018,
            None,
            "TV",
            80,
            ["Romance"],
        ),
        (
            "cyberpunk action movie before 2005",
            None,
            2004,
            "MOVIE",
            None,
            ["Action"],
        ),
        (
            "psychological thriller OVA 85+ score",
            None,
            None,
            "OVA",
            85,
            ["Psychological", "Thriller"],
        ),
        (
            "dark fantasy anime >2020 score:75+",
            2020,
            None,
            None,
            75,
            ["Fantasy"],
        ),
    ],
)
def test_rules_query_parsing(
    query,
    expected_year_min,
    expected_year_max,
    expected_format,
    expected_min_score,
    expected_genres,
):
    parsed = parse_query_rules(query)
    assert parsed.filters.year_min == expected_year_min
    assert parsed.filters.year_max == expected_year_max
    assert parsed.filters.format == expected_format
    assert parsed.filters.min_score == expected_min_score
    for g in expected_genres:
        assert g in parsed.filters.genres


def test_empty_query_fallback():
    parsed = parse_query_rules("TV")
    assert parsed.filters.format == "TV"
    assert parsed.semantic_text != ""  # Never empty string


# ===========================================================================
# Step 25: Domain Query Expander Tests
# ===========================================================================

def test_query_expander_military_politics():
    q = "dark political fantasy with military strategy and war"
    expanded = expand_query(q)
    assert q in expanded
    # Check that domain vocabulary was injected
    assert any(term in expanded.lower() for term in ["dystopian", "tragedy", "empire", "rebellion", "soldiers"])


def test_query_expander_cyberpunk():
    q = "cyberpunk psychological thriller"
    expanded = expand_query(q)
    assert q in expanded
    assert any(term in expanded.lower() for term in ["futuristic", "sci-fi", "neon", "technology", "dystopian"])


def test_query_expander_slice_of_life():
    q = "wholesome comedy slice of life"
    expanded = expand_query(q)
    assert any(term in expanded.lower() for term in ["healing", "heartwarming", "school life"])


# ===========================================================================
# Step 26: Unified LLM / Rules Parsing & Fallback Tests
# ===========================================================================

@pytest.mark.asyncio
async def test_unified_parser_without_api_key_falls_back():
    with patch("src.retrieval.llm_query_parser.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = ""
        result = await parse_query("90s action adventure anime series")
        assert isinstance(result, ParsedQuery)
        assert result.filters.year_min == 1990
        assert result.filters.year_max == 1999
        assert "Action" in result.filters.genres
        assert "Adventure" in result.filters.genres


@pytest.mark.asyncio
async def test_unified_parser_with_llm_success():
    mock_extracted = LLMExtractedQuery(
        semantic_mood="space western with bounty hunters and jazz vibes",
        year_min=1995,
        year_max=2000,
        format="TV",
        min_score=85,
        genres=["Action", "Sci-Fi"],
    )

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_extracted
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    with patch("src.retrieval.llm_query_parser.settings") as mock_settings, \
         patch("openai.AsyncOpenAI", return_value=mock_client):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"

        result = await parse_query("space cowboy anime from the late 90s score 85+")
        assert result.semantic_text == "space western with bounty hunters and jazz vibes"
        assert result.filters.year_min == 1995
        assert result.filters.year_max == 2000
        assert result.filters.format == "TV"
        assert result.filters.min_score == 85
