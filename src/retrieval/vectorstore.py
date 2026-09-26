"""ChromaDB Persistent Collection Setup & Vector Config (Step 22).

AnimeVectorStore manages the ChromaDB persistent client and the
``anime_catalog`` collection.  All vectors use **cosine** distance (HNSW).

Directory layout after initialisation::

    data/chroma/
    ├── chroma.sqlite3          ← collection registry
    └── <uuid>/                 ← HNSW index shards (auto-managed by ChromaDB)

Design decisions:
- ``anonymized_telemetry=False`` — no phone-home traffic.
- ``hnsw:space=cosine`` — chosen to match the normalised
  sentence-transformers output; inner-product would need L2-normalised vecs.
- ``get_or_create_collection`` — idempotent; safe to call on each startup.
- Collection name is configurable so tests can use an isolated namespace.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.core.config import settings
from src.core.logger import logger

# Collection name used across the codebase — change here only
COLLECTION_NAME: str = "anime_catalog"


class AnimeVectorStore:
    """Thin wrapper around a ChromaDB persistent collection.

    Provides collection access, count, reset, and upsert helpers used by
    the batch indexer (Step 23) and the retrieval searcher (Step 27).

    Args:
        collection_name: ChromaDB collection name (default ``anime_catalog``).
        persist_dir:     Override the ChromaDB persistence directory.
                         Defaults to ``settings.CHROMA_PERSIST_DIR``.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_dir: str | None = None,
    ) -> None:
        _dir = persist_dir or str(settings.CHROMA_PERSIST_DIR)
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaDB collection '{collection_name}' ready  "
            f"count={self.collection.count()}  path={_dir}"
        )

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of vectors currently in the collection."""
        return self.collection.count()

    def is_empty(self) -> bool:
        """Return True when no vectors have been indexed yet."""
        return self.count() == 0

    # ------------------------------------------------------------------
    # Upsert (used by Step 23 indexer)
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert a batch of vectors into the collection.

        ``upsert`` is idempotent: re-indexing an existing ID overwrites the
        stored vector, document, and metadata without creating duplicates.

        Args:
            ids:        Unique string ID per record (``AnimeDocument.doc_id``).
            embeddings: List of float vectors (one per document).
            documents:  Raw document text strings (stored for retrieval display).
            metadatas:  Flat dicts of scalar metadata (ChromaDB WHERE filters).
        """
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Query (used by Step 27 semantic searcher)
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 20,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """Run a nearest-neighbour query against the collection.

        Args:
            query_embedding: 384-dim (or 1536-dim) float vector.
            n_results:       Number of nearest neighbours to return.
            where:           Optional ChromaDB metadata filter dict.
            include:         Fields to return; defaults to metadatas + distances + documents.

        Returns:
            ChromaDB result dict with keys: ids, distances, metadatas, documents.
        """
        _include = include or ["metadatas", "distances", "documents"]
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": _include,
        }
        if where:
            kwargs["where"] = where

        return self.collection.query(**kwargs)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Delete and recreate the collection (destructive — use in tests only).

        Wipes all indexed vectors and resets the collection to empty.
        """
        name = self.collection.name
        logger.warning(f"Resetting ChromaDB collection '{name}' — all vectors will be deleted.")
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{name}' recreated (count=0).")

    def get_by_id(self, doc_id: str) -> dict | None:
        """Fetch a single document by its doc_id string.

        Args:
            doc_id: The ``AnimeDocument.doc_id`` string.

        Returns:
            Result dict with keys ids, embeddings, metadatas, documents,
            or None if not found.
        """
        result = self.collection.get(
            ids=[doc_id],
            include=["metadatas", "documents"],
        )
        if result and result.get("ids") and result["ids"]:
            return result
        return None
