"""
Abstract base interfaces for Embedding and LLM Providers (Step 03).
Defines contracts for BaseEmbedder and BaseChatModel to decouple provider implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseEmbedder(ABC):
    """
    Abstract interface for text embedding generation.
    Supports single query embedding and batch document embedding.
    """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding vector for a single query text.

        Args:
            text: Input query string.

        Returns:
            List of floats representing the dense embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a list of document strings.

        Args:
            texts: List of document strings.

        Returns:
            List of embedding vectors corresponding to the input texts.
        """
        pass


class BaseChatModel(ABC):
    """
    Abstract interface for LLM chat and generation models.
    Supports one-shot generation (with optional JSON schema) and asynchronous streaming.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_mode: bool = True,
    ) -> str:
        """
        Execute one-shot completion.

        Args:
            prompt: User or assembled context prompt.
            system_prompt: Optional system instruction prompt.
            json_mode: If True, instructs the model to adhere to strict JSON formatting.

        Returns:
            Generated response string.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream generated response tokens asynchronously for SSE / real-time UI.

        Args:
            prompt: User or assembled context prompt.
            system_prompt: Optional system instruction prompt.

        Returns:
            AsyncIterator yielding token strings as they arrive.
        """
        pass
