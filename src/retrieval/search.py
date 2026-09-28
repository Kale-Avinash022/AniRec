"""Semantic Vector Search with Metadata WHERE Clauses (Step 27).

Translates parsed :class:`ExtractedFilters` into ChromaDB ``where`` syntax,
embeds the semantic text, and runs a hybrid vector + filter query that
retrieves the top-K most relevant candidates.

Pipeline position::

    ParsedQuery
        │
        ▼
    build_chroma_filter()  →  ChromaDB WHERE dict
        │
        ▼
    search_candidates()    →  top-20 candidate dicts
        │                     each with: id, document, metadata, distance, similarity
        ▼
    [Step 28 relaxation if < 5 results]
        │
        ▼
    [Step 29 reranker]

ChromaDB field names used in WHERE clauses (from Step 18 metadata schema):
    season_year   int    → year_min / year_max filters
    format        str    → "TV", "MOVIE", "OVA", "ONA", "SPECIAL"
    average_score int    → min_score filter
    genres        str    → comma-joined; ChromaDB $contains for genre matching
"""

from __future__ import annotations

from typing import Any

from src.core.logger import logger
from src.retrieval.embedder import BaseEmbedder
from src.retrieval.query_expander import expand_query
from src.retrieval.query_parser import ExtractedFilters
from src.retrieval.vectorstore import AnimeVectorStore


# ---------------------------------------------------------------------------
# WHERE clause builder
# ---------------------------------------------------------------------------


def build_chroma_filter(filters: ExtractedFilters) -> dict[str, Any] | None:
    """Translate :class:`ExtractedFilters` into a ChromaDB ``where`` dict.

    Supports compound ``$and`` clauses when multiple constraints exist.
    Returns ``None`` when no filters are set (unfiltered search).

    Args:
        filters: Parsed query filters from Step 25/26.

    Returns:
        ChromaDB-compatible ``where`` dict, or ``None`` if filters are empty.

    Examples::

        build_chroma_filter(ExtractedFilters(year_min=2018, format="TV"))
        # → {"$and": [{"season_year": {"$gte": 2018}}, {"format": {"$eq": "TV"}}]}

        build_chroma_filter(ExtractedFilters(min_score=80))
        # → {"average_score": {"$gte": 80}}

        build_chroma_filter(ExtractedFilters())
        # → None
    """
    conditions: list[dict[str, Any]] = []

    if filters.year_min is not None:
        conditions.append({"season_year": {"$gte": filters.year_min}})
    if filters.year_max is not None:
        conditions.append({"season_year": {"$lte": filters.year_max}})
    if filters.format:
        conditions.append({"format": {"$eq": filters.format.upper()}})
    if filters.min_score is not None:
        conditions.append({"average_score": {"$gte": filters.min_score}})

    # Genre matching: genres stored as comma-joined string → use $contains
    # Note: ChromaDB $contains is substring match, which works for comma-joined genres
    for genre in filters.genres:
        conditions.append({"genres": {"$contains": genre}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------


def search_candidates(
    store: AnimeVectorStore,
    embedder: BaseEmbedder,
    semantic_text: str,
    filters: ExtractedFilters,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Embed semantic_text and run a filtered vector search against ChromaDB.

    Args:
        store:         :class:`AnimeVectorStore` instance.
        embedder:      :class:`BaseEmbedder` instance.
        semantic_text: Cleaned query text from the parser (no filter keywords).
        filters:       Parsed metadata constraints.
        top_k:         Number of candidates to retrieve (default 20).

    Returns:
        List of candidate dicts, each containing:
        ``id``, ``document``, ``metadata``, ``distance``, ``similarity``.
        Ordered by ascending distance (highest similarity first).
    """
    # Expand the query with anime-domain vocabulary before embedding
    # (fixes vocabulary mismatch: "dark" → "dystopian", "political" → "empire" etc.)
    expanded_text = expand_query(semantic_text)
    if expanded_text != semantic_text:
        logger.debug(f"Query expanded: '{semantic_text[:60]}' → '{expanded_text[:80]}'")

    query_vector = embedder.embed_query(expanded_text)
    chroma_where = build_chroma_filter(filters)

    try:
        results = store.query(
            query_embedding=query_vector,
            n_results=top_k,
            where=chroma_where,
            include=["metadatas", "distances", "documents"],
        )
    except Exception as exc:
        # ChromaDB raises if WHERE results in 0 matching docs before vector search
        logger.warning(f"ChromaDB query error (filter may be too strict): {exc}")
        return []

    candidates: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]

    for doc_id, dist, meta, doc_text in zip(ids, distances, metadatas, documents):
        # ChromaDB cosine distance d in [0, 2] -> similarity = 1 - d
        # For L2 distance or safety: bounded in [0.0, 1.0]
        if dist <= 2.0:
            sim = max(0.0, min(1.0, 1.0 - dist))
        else:
            sim = 1.0 / (1.0 + max(0.0, dist))

        sim_rounded = round(float(sim), 4)
        candidates.append(
            {
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
                "distance": dist,
                "similarity": sim_rounded,
                "similarity_score": sim_rounded,
            }
        )

    logger.debug(
        f"search_candidates: '{semantic_text[:60]}' → {len(candidates)} results "
        f"filter={chroma_where}"
    )
    return candidates


def retrieve_title_candidate(
    store: AnimeVectorStore,
    embedder: BaseEmbedder,
    reranker: Any,
    title: str,
) -> dict[str, Any] | None:
    """Retrieve the single most relevant candidate anime matching a specific title.

    Args:
        store: :class:`AnimeVectorStore` instance.
        embedder: :class:`BaseEmbedder` instance.
        reranker: :class:`Reranker` instance.
        title: Show title string (e.g. "Eden Zero", "Fairy Tail").

    Returns:
        Top matching candidate dict or None.
    """
    candidates = search_candidates(
        store=store,
        embedder=embedder,
        semantic_text=title,
        filters=ExtractedFilters(),
        top_k=12,
    )
    if not candidates:
        return None

    title_lower = title.lower().strip()
    # Check exact title match first
    for c in candidates:
        meta = c.get("metadata", {})
        eng = (meta.get("title_english") or "").lower()
        rom = (meta.get("title_romaji") or "").lower()
        if (eng and eng == title_lower) or (rom and rom == title_lower):
            return c

    # Check substring containment
    for c in candidates:
        meta = c.get("metadata", {})
        eng = (meta.get("title_english") or "").lower()
        rom = (meta.get("title_romaji") or "").lower()
        if (eng and title_lower in eng) or (rom and title_lower in rom):
            return c

    # Fallback to cross-encoder reranking
    try:
        rescored = reranker.rerank(query=title, candidates=candidates, top_k=3)
        if rescored:
            return rescored[0]
    except Exception as exc:
        logger.warning(f"Reranking error during title candidate lookup for '{title}': {exc}")

    return candidates[0]

