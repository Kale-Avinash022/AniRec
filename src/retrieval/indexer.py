"""Batch Indexing Pipeline with Upsert & Progress Tracking (Step 23).

Loads the processed corpus (data/processed/anime_corpus.json), embeds every
AnimeDocument's ``document_text`` in batches, and upserts the resulting
vectors into the ChromaDB ``anime_catalog`` collection.

Upsert semantics guarantee idempotency:
  - Re-running the indexer overwrites existing records (no duplicates).
  - Interrupted runs can be safely re-run — the embedding is deterministic.

Usage:
    python scripts/run_indexer.py
    python scripts/run_indexer.py --batch-size 128 --force-reindex
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from tqdm import tqdm

from src.core.logger import logger
from src.docs.snapshot import load_processed_corpus
from src.retrieval.embedder import get_embedder
from src.retrieval.vectorstore import AnimeVectorStore

if TYPE_CHECKING:
    from src.docs.schema import AnimeDocument
    from src.retrieval.embedder import BaseEmbedder


# ---------------------------------------------------------------------------
# Core indexer
# ---------------------------------------------------------------------------


def build_vector_index(
    batch_size: int = 64,
    force_reindex: bool = False,
    store: AnimeVectorStore | None = None,
    embedder: "BaseEmbedder | None" = None,
) -> int:
    """Embed and upsert all AnimeDocuments into ChromaDB.

    Args:
        batch_size:    Number of documents to embed and upsert per iteration.
                       64 is optimal for CPU; increase to 128 on GPU.
        force_reindex: If True, resets the collection before indexing.
                       If False (default), skips if already fully indexed.
        store:         Override AnimeVectorStore (injected in tests).
        embedder:      Override embedder (injected in tests).

    Returns:
        Total number of vectors in the collection after indexing.

    Raises:
        FileNotFoundError: If the processed corpus doesn't exist (run
                           ``scripts/run_pipeline.py`` first).
    """
    _store = store or AnimeVectorStore()
    _embedder = embedder or get_embedder()

    # ── Load corpus ─────────────────────────────────────────────────────────
    logger.info("Loading processed corpus from data/processed/ ...")
    corpus: list[AnimeDocument] = load_processed_corpus()
    total = len(corpus)
    logger.info(f"Corpus loaded: {total} AnimeDocuments")

    # ── Skip check ──────────────────────────────────────────────────────────
    if not force_reindex and _store.count() >= total:
        logger.info(
            f"Collection already has {_store.count()} vectors (≥ corpus size {total}). "
            "Skipping indexing. Pass force_reindex=True to re-embed."
        )
        return _store.count()

    if force_reindex:
        logger.warning("force_reindex=True — resetting collection before indexing.")
        _store.reset()

    # ── Batch embedding + upsert loop ────────────────────────────────────────
    num_batches = math.ceil(total / batch_size)
    t_start = time.perf_counter()

    logger.info(
        f"Starting indexing: {total} docs  batch_size={batch_size}  "
        f"num_batches={num_batches}"
    )

    indexed = 0
    with tqdm(total=total, desc="Indexing", unit="doc", ncols=80) as pbar:
        for batch_start in range(0, total, batch_size):
            batch: list[AnimeDocument] = corpus[batch_start : batch_start + batch_size]

            # Prepare ChromaDB upsert fields
            ids: list[str] = [doc.doc_id for doc in batch]
            documents: list[str] = [doc.document_text for doc in batch]
            metadatas: list[dict] = [doc.metadata for doc in batch]

            # Generate embeddings for this batch
            embeddings: list[list[float]] = _embedder.embed_documents(documents)

            # Upsert into ChromaDB (idempotent)
            _store.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            indexed += len(batch)
            pbar.update(len(batch))

    elapsed = time.perf_counter() - t_start
    final_count = _store.count()

    logger.info("=" * 60)
    logger.info("✅  Indexing complete")
    logger.info(f"   Docs indexed  : {indexed}")
    logger.info(f"   Collection cnt: {final_count}")
    logger.info(f"   Elapsed       : {elapsed:.1f}s  ({total / elapsed:.0f} docs/sec)")
    logger.info("=" * 60)

    return final_count
