"""Tests for Structured LLM Query Extraction & Graceful Fallback (Step 26)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.retrieval.llm_query_parser import (
    LLMExtractedQuery,
    parse_query_with_llm,
    parse_query,
)
from src.retrieval.query_parser import ParsedQuery, ExtractedFilters


def test_llm_extracted_query_schema():
    model = LLMExtractedQuery(
        semantic_mood="cyberpunk thriller in neo tokyo",
        year_min=1990,
        year_max=1999,
        format="TV",
        min_score=80,
        genres=["Action", "Sci-Fi"],
    )
    assert model.year_min == 1990
    assert model.year_max == 1999
    assert model.format == "TV"
    assert model.min_score == 80
    assert model.genres == ["Action", "Sci-Fi"]


@pytest.mark.asyncio
async def test_fallback_when_api_key_unset():
    with patch("src.retrieval.llm_query_parser.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = ""
        result = await parse_query_with_llm("90s action anime score:80+")
        assert result is None


@pytest.mark.asyncio
async def test_unified_parse_query_falls_back_to_rules():
    with patch("src.retrieval.llm_query_parser.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = ""
        result = await parse_query("wholesome romance TV anime >2015 score:75+")
        assert isinstance(result, ParsedQuery)
        assert result.filters.format == "TV"
        assert result.filters.year_min == 2015
        assert result.filters.min_score == 75


@pytest.mark.asyncio
async def test_successful_mocked_llm_extraction():
    mock_extracted = LLMExtractedQuery(
        semantic_mood="philosophical giant robot battles with complex politics",
        year_min=1990,
        year_max=1999,
        format="TV",
        min_score=80,
        genres=["Mecha", "Drama"],
    )

    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_extracted

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

    with patch("src.retrieval.llm_query_parser.settings") as mock_settings, \
         patch("openai.AsyncOpenAI", return_value=mock_client):
        mock_settings.OPENAI_API_KEY = "sk-fake-key-for-test"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"

        result = await parse_query_with_llm("90s deep mecha anime with high rating")
        assert result is not None
        assert result.semantic_text == "philosophical giant robot battles with complex politics"
        assert result.filters.year_min == 1990
        assert result.filters.year_max == 1999
        assert result.filters.format == "TV"
        assert result.filters.genres == ["Mecha", "Drama"]
