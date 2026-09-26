"""Tests for Abstract LLM Provider Interface & Implementations (Step 32)."""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from src.generation.llm_client import (
    BaseChatModel,
    MockChatModel,
    OllamaChatModel,
    OpenAIChatModel,
    get_chat_model,
)


@pytest.mark.asyncio
async def test_mock_chat_model_generate():
    custom_resp = json.dumps({"status": "ok", "msg": "test"})
    model = MockChatModel(response_text=custom_resp)
    res = await model.generate("prompt", system_prompt="system", json_mode=True)
    assert res == custom_resp


@pytest.mark.asyncio
async def test_mock_chat_model_stream():
    model = MockChatModel(response_text="Hello world test")
    tokens = []
    async for t in model.generate_stream("prompt"):
        tokens.append(t)
    reconstructed = "".join(tokens)
    assert reconstructed == "Hello world test"


def test_get_chat_model_factory():
    # Explicit mock
    mock_model = get_chat_model(provider="mock")
    assert isinstance(mock_model, MockChatModel)

    # Fallback to mock when openai key missing
    with patch("src.generation.llm_client.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.OPENAI_API_KEY = ""
        model = get_chat_model()
        assert isinstance(model, MockChatModel)


@pytest.mark.asyncio
async def test_openai_chat_model_generate():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"recommendations": []}'
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        model = OpenAIChatModel(api_key="fake-key", model="gpt-4o-mini")
        res = await model.generate(prompt="hello", system_prompt="system")
        assert res == '{"recommendations": []}'
        mock_client.chat.completions.create.assert_called_once()
