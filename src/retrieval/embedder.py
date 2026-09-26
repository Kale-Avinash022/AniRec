"""Abstract Embedder Interface & Concrete Implementations (Step 21).

Provider abstraction layer for text embedding generation.
Switches between local SentenceTransformers and OpenAI Cloud via config.

Hierarchy:
    BaseEmbedder (src.core.interfaces)
        ├── SentenceTransformerEmbedder  ← default (zero-cost, local, fast)
        └── OpenAIEmbedder               ← cloud option (set EMBEDDING_PROVIDER=openai)

Usage:
    from src.retrieval.embedder import get_embedder

    embedder = get_embedder()                           # reads settings.EMBEDDING_PROVIDER
    vec = embedder.embed_query("dark fantasy anime")    # → list[float] (384-dim)
    vecs = embedder.embed_documents(["text1", ...])     # → list[list[float]]
"""

from __future__ import annotations

from src.core.config import settings
from src.core.interfaces import BaseEmbedder
from src.core.logger import logger


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding using sentence-transformers (zero cost, privacy-first).

    Default model: ``all-MiniLM-L6-v2``
      - 384 dimensions
      - 256 token sequence window (hard truncation at 512)
      - ~180 docs/sec on CPU
      - 3,000 anime docs ≈ 4.6 MB raw float32 footprint

    The model is downloaded once on first init and cached in
    ``~/.cache/huggingface/hub``.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        _model_name = model_name or settings.LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading SentenceTransformer model: {_model_name}")
        self.model = SentenceTransformer(_model_name)
        self.model_name = _model_name
        dim = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension")
        logger.info(f"SentenceTransformerEmbedder ready  dim={dim()}")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Raw query text (e.g. ``"dark fantasy with political intrigue"``).

        Returns:
            384-dimensional float vector.
        """
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of document strings.

        Args:
            texts: List of document strings (e.g. assembled anime documents).

        Returns:
            List of 384-dimensional float vectors, one per input string.
        """
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    @property
    def embedding_dim(self) -> int:
        """Return the output dimension of this model."""
        dim = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension")
        return dim()


class OpenAIEmbedder(BaseEmbedder):
    """Cloud embedding via OpenAI Embeddings API.

    Default model: ``text-embedding-3-small``
      - 1536 dimensions (or 512 when dimension is specified)
      - 8191 token window
      - Cost: $0.02 / 1M tokens

    Requires ``OPENAI_API_KEY`` in ``.env``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from openai import OpenAI

        _key = api_key or settings.OPENAI_API_KEY
        if not _key:
            raise ValueError(
                "OpenAIEmbedder requires OPENAI_API_KEY. "
                "Set it in .env or pass api_key= explicitly."
            )
        self.client = OpenAI(api_key=_key)
        self.model_name = model or settings.OPENAI_EMBEDDING_MODEL
        logger.info(f"OpenAIEmbedder ready  model={self.model_name}")

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string via OpenAI API.

        Args:
            text: Query text.

        Returns:
            1536-dimensional float vector.
        """
        resp = self.client.embeddings.create(input=[text], model=self.model_name)
        return resp.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed documents via OpenAI API.

        OpenAI accepts up to 2048 strings per call; caller should chunk if needed.

        Args:
            texts: List of document strings.

        Returns:
            List of 1536-dimensional float vectors.
        """
        resp = self.client.embeddings.create(input=texts, model=self.model_name)
        # OpenAI guarantees order matches input
        return [item.embedding for item in resp.data]

    @property
    def embedding_dim(self) -> int:
        """Return 1536 (default for text-embedding-3-small)."""
        return 1536


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder() -> BaseEmbedder:
    """Return the configured embedder based on ``settings.EMBEDDING_PROVIDER``.

    - ``"local"``  → :class:`SentenceTransformerEmbedder` (default, free, offline)
    - ``"openai"`` → :class:`OpenAIEmbedder` (cloud, requires API key)

    Returns:
        A ready-to-use :class:`BaseEmbedder` instance.
    """
    provider = settings.EMBEDDING_PROVIDER
    if provider == "openai":
        logger.info("Embedding provider: OpenAI")
        return OpenAIEmbedder()
    logger.info("Embedding provider: local SentenceTransformer")
    return SentenceTransformerEmbedder()
