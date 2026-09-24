"""
Unit tests for Step 03: Provider Abstractions & Interface Contracts.
"""

from collections.abc import AsyncIterator

import pytest

from src.core.interfaces import BaseChatModel, BaseEmbedder


def test_base_embedder_cannot_be_instantiated_directly():
    """Verify that BaseEmbedder enforces ABC rules."""
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_base_chat_model_cannot_be_instantiated_directly():
    """Verify that BaseChatModel enforces ABC rules."""
    with pytest.raises(TypeError):
        BaseChatModel()


class DummyEmbedder(BaseEmbedder):
    """Concrete dummy implementation for testing contract satisfaction."""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class DummyChatModel(BaseChatModel):
    """Concrete dummy implementation for testing contract satisfaction."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
    ) -> str:
        return '{"response": "ok"}'

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        async def _stream():
            for token in ["Hello", " ", "World"]:
                yield token

        return _stream()


def test_dummy_embedder_methods():
    """Verify dummy embedder complies with interface."""
    embedder = DummyEmbedder()
    query_vec = embedder.embed_query("test query")
    assert isinstance(query_vec, list)
    assert len(query_vec) == 3

    docs_vec = embedder.embed_documents(["doc1", "doc2"])
    assert len(docs_vec) == 2
    assert docs_vec[0] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_dummy_chat_model_methods():
    """Verify dummy chat model complies with interface."""
    chat_model = DummyChatModel()
    response = await chat_model.generate("test prompt")
    assert response == '{"response": "ok"}'

    stream = await chat_model.generate_stream("test prompt")
    tokens = [token async for token in stream]
    assert "".join(tokens) == "Hello World"
