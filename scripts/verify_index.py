"""Vector Index Sanity Verification Script (Step 24).

Runs a set of curated test queries against the live ChromaDB index and
prints the top-5 nearest-neighbour results for each.  Used to confirm:

  1. The index is populated and cosine search works.
  2. Returned titles are semantically related to the query.
  3. Similarity scores fall in the expected range (0.60 – 0.95).
  4. Metadata filtering (WHERE clauses) returns correct subsets.

Usage:
    python scripts/verify_index.py
    python scripts/verify_index.py --top-k 10
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.core.logger import logger
from src.retrieval.embedder import get_embedder
from src.retrieval.query_expander import expand_query
from src.retrieval.vectorstore import AnimeVectorStore


# ---------------------------------------------------------------------------
# Verification queries
# ---------------------------------------------------------------------------

SANITY_QUERIES: list[dict] = [
    {
        "query": "dark political fantasy with military strategy and war",
        "expect_genres": ["Action", "Drama"],
    },
    {
        "query": "cyberpunk psychological thriller in futuristic Tokyo",
        "expect_genres": ["Sci-Fi"],
    },
    {
        "query": "wholesome high school romance with comedy slice of life",
        "expect_genres": ["Romance", "Comedy"],
    },
    {
        "query": "space bounty hunters with comedic adventures",
        "expect_genres": ["Action", "Sci-Fi"],
    },
    {
        "query": "magical girl transformation battles with friendship themes",
        "expect_genres": ["Mahou Shoujo"],
    },
]


# ---------------------------------------------------------------------------
# Core verification function
# ---------------------------------------------------------------------------


def verify_index(top_k: int = 5) -> bool:
    """Run all sanity queries and report results.

    Args:
        top_k: Number of results to retrieve per query.

    Returns:
        True if all queries returned results; False if any returned zero.
    """
    store = AnimeVectorStore()
    embedder = get_embedder()

    total = store.count()
    if total == 0:
        logger.error(
            "Collection is empty! Run 'python scripts/run_indexer.py' first."
        )
        return False

    logger.info(f"ChromaDB collection: {total} vectors  | top_k={top_k}")
    logger.info("=" * 60)

    all_passed = True

    for spec in SANITY_QUERIES:
        query_text: str = spec["query"]
        expanded = expand_query(query_text)
        logger.info(f"\n🔍  Query: \"{query_text}\"")
        if expanded != query_text:
            logger.info(f"   ↳ Expanded: \"{expanded[:100]}\"")

        query_vec = embedder.embed_query(expanded)
        results = store.query(
            query_embedding=query_vec,
            n_results=top_k,
            include=["metadatas", "distances", "documents"],
        )

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not ids:
            logger.warning("  ⚠  No results returned!")
            all_passed = False
            continue

        for rank, (doc_id, dist, meta) in enumerate(
            zip(ids, distances, metadatas), start=1
        ):
            # ChromaDB cosine distance is 1 - cosine_similarity
            similarity = round(1.0 - dist, 4)
            title = meta.get("title_romaji", meta.get("title_english", "?"))
            year = meta.get("season_year", "?")
            genres = meta.get("genres", "")
            score = meta.get("average_score", "?")
            fmt = meta.get("format", "?")

            logger.info(
                f"  {rank}. [{similarity:.3f}] {title} ({year}) "
                f"| {fmt} | score={score} | genres={genres}"
            )

        # Warn if top similarity is suspiciously low
        top_sim = 1.0 - distances[0] if distances else 0.0
        if top_sim < 0.4:
            logger.warning(
                f"  ⚠  Top similarity {top_sim:.3f} is low — "
                "check embedding model or document quality."
            )

    logger.info("\n" + "=" * 60)
    status = "✅  ALL QUERIES PASSED" if all_passed else "❌  SOME QUERIES RETURNED NO RESULTS"
    logger.info(status)
    return all_passed


# ---------------------------------------------------------------------------
# Filtered verification helper
# ---------------------------------------------------------------------------


def verify_with_filter(
    query_text: str,
    where: dict,
    top_k: int = 5,
) -> None:
    """Run a single query with a ChromaDB metadata WHERE filter and print results.

    Example::

        verify_with_filter(
            "romance in high school",
            where={"season_year": {"$gte": 2020}},
        )

    Args:
        query_text: Natural language query.
        where:      ChromaDB filter dict (e.g. ``{"format": "TV"}``).
        top_k:      Number of results to retrieve.
    """
    store = AnimeVectorStore()
    embedder = get_embedder()

    logger.info(f"🔍  Filtered Query: \"{query_text}\"")
    logger.info(f"   WHERE: {where}")

    query_vec = embedder.embed_query(query_text)
    results = store.query(
        query_embedding=query_vec,
        n_results=top_k,
        where=where,
        include=["metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        logger.warning("  No results for this filter combination.")
        return

    for rank, (dist, meta) in enumerate(zip(distances, metadatas), start=1):
        sim = round(1.0 - dist, 4)
        title = meta.get("title_romaji", "?")
        year = meta.get("season_year", "?")
        logger.info(f"  {rank}. [{sim:.3f}] {title} ({year})")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AniRec vector index sanity check")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query (default 5)")
    args = parser.parse_args()

    passed = verify_index(top_k=args.top_k)

    # Bonus: demonstrate a filtered query
    logger.info("\n📌  Bonus — filtered search (TV format, score ≥ 80, year ≥ 2015):")
    verify_with_filter(
        "action adventure with strong world-building",
        where={
            "$and": [
                {"format": {"$eq": "TV"}},
                {"average_score": {"$gte": 80}},
                {"season_year": {"$gte": 2015}},
            ]
        },
        top_k=5,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
