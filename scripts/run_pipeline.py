"""End-to-End AniRec Data Pipeline Runner (Steps 08–19).

Runs the full pipeline:
  Steps 11    → Fetch anime pages from AniList (disk cache + retry + rate limit)
  Steps 12-14 → Transform, franchise graph, Pydantic validation  [run_full_pipeline]
  Steps 15-18 → Document text assembly + metadata schema          [build_document_batch]
  Step 19     → Persist to data/processed/anime_corpus.json       [save_processed_corpus]

Usage:
    python scripts/run_pipeline.py                   # full catalog (~3,000 shows)
    python scripts/run_pipeline.py --max-pages 5     # quick smoke test (250 shows)
    python scripts/run_pipeline.py --no-cache        # force re-fetch from AniList
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.core.logger import logger
from src.docs.assembler import build_document_batch
from src.docs.snapshot import get_corpus_stats, save_processed_corpus
from src.ingest import (
    AniListClient,
    SanitizedAnime,
    build_franchise_graph,
    build_root_id_map,
    run_full_pipeline,
    sanitize_raw_media_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AniRec end-to-end data pipeline")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=settings.ANILIST_MAX_PAGES,
        help=f"Max pages to fetch (default {settings.ANILIST_MAX_PAGES}; 1 page = 50 shows)",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=settings.ANILIST_PER_PAGE,
        help="Results per page (max 50, AniList hard limit)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass disk cache and force re-fetch from AniList API",
    )
    parser.add_argument(
        "--min-tag-rank",
        type=int,
        default=40,
        help="Minimum community tag rank to include in document text (default 40)",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=15,
        help="Max tags to include per document (default 15)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()

    # Ensure all data/ subdirectories exist
    settings.ensure_directories()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1  — Fetch raw anime pages from AniList  (Step 11)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 / STEP 11: AniList GraphQL Ingestion")
    logger.info(
        f"  max_pages={args.max_pages}  per_page={args.per_page}  "
        f"cache={'OFF' if args.no_cache else 'ON'}"
    )
    logger.info("=" * 60)

    client = AniListClient()
    raw_media_list = await client.ingest_catalog(
        max_pages=args.max_pages,
        per_page=args.per_page,
        use_cache=not args.no_cache,
    )
    logger.info(f"✅  Stage 1 done → {len(raw_media_list)} raw GraphQL media records")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2  — Transform → Franchise Graph → Validation  (Steps 12-14)
    #            run_full_pipeline chains all three in one call
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 / STEPS 12-14: Transform, Franchise Graph & Validation")
    logger.info("=" * 60)

    enriched_dicts, report = run_full_pipeline(raw_media_list)

    logger.info(str(report))
    logger.info(
        f"✅  Stage 2 done → {len(enriched_dicts)} enriched records | "
        f"health={report.data_health_score}%"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3  — Re-hydrate to SanitizedAnime so assembler is type-safe
    #            (enriched_dicts already have franchise_root_id injected)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3: Re-hydrate enriched dicts → SanitizedAnime + root_map")
    logger.info("=" * 60)

    catalog: list[SanitizedAnime] = []
    root_map: dict[int, int] = {}
    for d in enriched_dicts:
        try:
            anime = SanitizedAnime.model_validate(d)
            catalog.append(anime)
            root_map[anime.id] = d.get("franchise_root_id", anime.id)
        except Exception as exc:
            logger.debug(f"Re-hydration skip id={d.get('id')}: {exc}")

    logger.info(f"✅  Stage 3 done → {len(catalog)} SanitizedAnime objects ready")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4  — Assemble document text + metadata (Steps 15–18)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 / STEPS 15-18: Document Text & Metadata Assembly")
    logger.info(f"  min_tag_rank={args.min_tag_rank}  max_tags={args.max_tags}")
    logger.info("=" * 60)

    documents, assembly_errors = build_document_batch(
        catalog=catalog,
        root_map=root_map,
        min_tag_rank=args.min_tag_rank,
        max_tags=args.max_tags,
    )

    logger.info(
        f"✅  Stage 4 done → {len(documents)} AnimeDocuments "
        f"({len(assembly_errors)} assembly errors)"
    )

    # Print one sample document so you can visually verify the format
    if documents:
        sample = documents[0]
        logger.info("\n" + "─" * 50)
        logger.info(f"SAMPLE — id={sample.anime_id}  franchise_root={sample.franchise_root_id}")
        logger.info(sample.document_text[:600])
        logger.info("─" * 50 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5  — Persist to data/processed/  (Step 19)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 / STEP 19: Persisting Corpus to data/processed/")
    logger.info("=" * 60)

    output_path = save_processed_corpus(documents)
    stats = get_corpus_stats()

    elapsed = time.perf_counter() - t_start

    logger.info("=" * 60)
    logger.info("🎉  PIPELINE COMPLETE")
    logger.info(f"  Output path    : {output_path}")
    logger.info(f"  Records saved  : {stats['record_count']}")
    logger.info(f"  Schema version : {stats['schema_version']}")
    logger.info(f"  Elapsed time   : {elapsed:.1f}s")
    logger.info("=" * 60)
    logger.info("")
    logger.info("▶  NEXT: Phase 4 — Embed & Index into ChromaDB (Steps 20–24)")
    logger.info("   Run: python scripts/run_indexer.py")


if __name__ == "__main__":
    asyncio.run(main())
