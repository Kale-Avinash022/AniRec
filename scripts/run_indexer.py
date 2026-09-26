"""AniRec ChromaDB Indexer Runner (Steps 20–23).

Embeds all 3,000 AnimeDocuments from data/processed/anime_corpus.json
and upserts them into the ChromaDB persistent collection.

Usage:
    python scripts/run_indexer.py                   # incremental (skips if up-to-date)
    python scripts/run_indexer.py --batch-size 128  # larger batches (faster on GPU)
    python scripts/run_indexer.py --force-reindex   # wipe & re-embed everything
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logger import logger
from src.retrieval.indexer import build_vector_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AniRec ChromaDB batch indexer")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Docs per embedding batch (default 64; try 128 on GPU)",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Reset collection and re-embed all documents from scratch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("AniRec Phase 4 — ChromaDB Indexer")
    logger.info(f"  batch_size={args.batch_size}  force_reindex={args.force_reindex}")
    logger.info("=" * 60)

    final_count = build_vector_index(
        batch_size=args.batch_size,
        force_reindex=args.force_reindex,
    )

    logger.info(f"🎉  Index ready: {final_count} vectors in ChromaDB")
    logger.info("")
    logger.info("▶  NEXT: Verify the index with:")
    logger.info("   python scripts/verify_index.py")
    logger.info("")
    logger.info("▶  THEN: Phase 5 — Hybrid Retrieval & Reranking (Steps 25–31)")


if __name__ == "__main__":
    main()
