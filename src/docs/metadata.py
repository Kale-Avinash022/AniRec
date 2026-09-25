"""Metadata Schema Construction for Vector Filtering (Step 18).

What this module does
---------------------
ChromaDB stores a flat metadata dict alongside every vector.  This dict is
the only mechanism for **exact-match and range filtering** during retrieval
(e.g. ``year >= 2015``, ``format == "TV"``, ``average_score >= 80``).

Getting this schema wrong silently breaks all downstream filtering:
- Wrong type (list instead of str) → ChromaDB silently ignores the field
- Missing field → WHERE clause throws KeyError at query time
- Inconsistent naming → Step 25/27 filter parsers can't locate fields

This module owns the **single source of truth** for:
1. The exact field names and types stored in every ChromaDB document.
2. The valid operators for each field (used by Steps 25–27 query parsing).
3. A builder function that always produces a type-safe, schema-compliant dict.
4. A WHERE-clause assembler for composing ChromaDB filter conditions.

ChromaDB metadata type constraints
-----------------------------------
Values must be one of: ``str``, ``int``, ``float``, ``bool``.
Lists, dicts, None, and nested objects are **not** supported.

  - Multi-value fields (genres, tags) are stored as comma-joined strings.
  - Nullable int/float fields fall back to ``0`` (never ``None``).
  - Nullable str fields fall back to ``""`` (never ``None``).
"""

from __future__ import annotations

from typing import Any

from src.docs.tags import (
    compute_tag_density_score,
    get_top_tag_names,
)
from src.ingest.transform import SanitizedAnime

# ---------------------------------------------------------------------------
# Schema field registry — single source of truth for field names & types
# ---------------------------------------------------------------------------

#: Mapping of metadata field name → Python type.
#: Referenced by Step 25 (rules-based filter parser) and Step 27 (retrieval).
METADATA_FIELD_TYPES: dict[str, type] = {
    # Identifiers
    "anime_id": int,
    "id_mal": int,
    "franchise_root_id": int,
    # Titles
    "title_romaji": str,
    "title_english": str,
    # Temporal
    "season_year": int,
    "season": str,           # WINTER | SPRING | SUMMER | FALL | ""
    # Classification
    "format": str,           # TV | MOVIE | OVA | ONA | SPECIAL | MUSIC
    "status": str,           # FINISHED | RELEASING | NOT_YET_RELEASED | CANCELLED
    "episodes": int,
    # Taxonomy (comma-joined — split on "," to recover lists)
    "genres": str,
    "top_tags": str,
    # Scores
    "average_score": int,    # 0–100 (0 = unrated)
    "mean_score": int,
    "popularity": int,
    "favourites": int,
    # Quality signal
    "tag_density": float,    # 0.0–1.0
    # Production
    "studio": str,
    # Assets
    "cover_image_url": str,
    "site_url": str,
}

#: Valid ChromaDB ``where`` operators per field type.
#: Used by Step 25 (query parser) to validate filter conditions.
FIELD_OPERATORS: dict[str, list[str]] = {
    "int": ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"],
    "float": ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"],
    "str": ["$eq", "$ne"],
}

#: Fields that support range filters (int/float).
RANGE_FILTER_FIELDS: frozenset[str] = frozenset(
    k for k, v in METADATA_FIELD_TYPES.items() if v in (int, float)
)

#: Fields that support equality filters (str).
EQUALITY_FILTER_FIELDS: frozenset[str] = frozenset(
    k for k, v in METADATA_FIELD_TYPES.items() if v is str
)


# ---------------------------------------------------------------------------
# Core metadata builder
# ---------------------------------------------------------------------------


def build_vector_metadata(
    anime: SanitizedAnime,
    franchise_root_id: int,
    min_tag_rank: int = 40,
) -> dict[str, Any]:
    """Build a ChromaDB-compatible flat metadata dict for a single anime.

    Every value is guaranteed to be ``str``, ``int``, or ``float`` — no
    ``None``, no lists, no nested dicts.  Field names exactly match
    ``METADATA_FIELD_TYPES``.

    Args:
        anime:             ``SanitizedAnime`` from Step 12.
        franchise_root_id: From ``build_root_id_map()`` in Step 13.
        min_tag_rank:      Minimum rank for tag inclusion (default 40).

    Returns:
        Flat metadata dict ready for ChromaDB upsert.
    """
    top_tag_names = get_top_tag_names(anime.tags, n=5, min_rank=min_tag_rank)
    tag_density = compute_tag_density_score(anime.tags, min_rank=min_tag_rank)

    return {
        # ── Identifiers ────────────────────────────────────────────────────
        "anime_id": int(anime.id),
        "id_mal": int(anime.id_mal) if anime.id_mal is not None else 0,
        "franchise_root_id": int(franchise_root_id),
        # ── Titles ─────────────────────────────────────────────────────────
        "title_romaji": str(anime.title_romaji),
        "title_english": str(anime.title_english or ""),
        # ── Temporal ───────────────────────────────────────────────────────
        "season_year": int(anime.season_year) if anime.season_year is not None else 0,
        "season": str(anime.season or ""),
        # ── Classification ─────────────────────────────────────────────────
        "format": str(anime.format or ""),
        "status": str(anime.status or ""),
        "episodes": int(anime.episodes) if anime.episodes is not None else 0,
        # ── Taxonomy ───────────────────────────────────────────────────────
        # Stored as comma-joined strings — ChromaDB does not support lists
        "genres": ",".join(anime.genres) if anime.genres else "",
        "top_tags": ",".join(top_tag_names),
        # ── Scores ─────────────────────────────────────────────────────────
        "average_score": int(anime.average_score) if anime.average_score is not None else 0,
        "mean_score": int(anime.mean_score) if anime.mean_score is not None else 0,
        "popularity": int(anime.popularity),
        "favourites": int(anime.favourites),
        # ── Quality signal ─────────────────────────────────────────────────
        "tag_density": float(tag_density),
        # ── Production ─────────────────────────────────────────────────────
        "studio": str(anime.studios[0]) if anime.studios else "",
        # ── Assets ─────────────────────────────────────────────────────────
        "cover_image_url": str(anime.cover_image_large or ""),
        "site_url": str(anime.site_url or ""),
    }


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """Validate a metadata dict against the schema and return any violations.

    Does NOT raise — returns a list of violation strings so the caller can
    decide whether to log, skip, or hard-fail.

    Args:
        metadata: Dict produced by ``build_vector_metadata()``.

    Returns:
        List of violation strings.  Empty list means the dict is valid.
    """
    violations: list[str] = []

    for field, expected_type in METADATA_FIELD_TYPES.items():
        if field not in metadata:
            violations.append(f"Missing field: '{field}'")
            continue

        value = metadata[field]
        if value is None:
            violations.append(f"Field '{field}' is None — must be {expected_type.__name__}")
        elif not isinstance(value, expected_type):
            violations.append(
                f"Field '{field}' has type {type(value).__name__!r}, "
                f"expected {expected_type.__name__!r} (value={value!r})"
            )

    # Check for extra fields not in schema
    for key in metadata:
        if key not in METADATA_FIELD_TYPES:
            violations.append(f"Extra field not in schema: '{key}'")

    return violations


# ---------------------------------------------------------------------------
# WHERE-clause builder (used by Steps 25–27)
# ---------------------------------------------------------------------------


def build_where_clause(
    format_filter: str | None = None,
    min_score: int | None = None,
    max_score: int | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    status_filter: str | None = None,
    franchise_root_id: int | None = None,
    exclude_franchise_root_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    """Compose a ChromaDB-compatible ``where`` clause from structured filter params.

    Returns ``None`` when no filters are active (ChromaDB requires ``None``
    rather than ``{}`` for an unfiltered query).

    Args:
        format_filter:              e.g. ``"TV"``, ``"MOVIE"``
        min_score:                  Inclusive lower bound for ``average_score``.
        max_score:                  Inclusive upper bound for ``average_score``.
        min_year:                   Inclusive lower bound for ``season_year``.
        max_year:                   Inclusive upper bound for ``season_year``.
        status_filter:              e.g. ``"FINISHED"``, ``"RELEASING"``
        franchise_root_id:          Restrict to a single franchise cluster.
        exclude_franchise_root_ids: Exclude specific franchise clusters.

    Returns:
        ChromaDB ``where`` dict, or ``None`` if no filters provided.

    Examples:
        >>> build_where_clause(format_filter="TV", min_score=80, min_year=2015)
        {'$and': [{'format': {'$eq': 'TV'}}, {'average_score': {'$gte': 80}},
                  {'season_year': {'$gte': 2015}}]}
    """
    conditions: list[dict[str, Any]] = []

    if format_filter:
        conditions.append({"format": {"$eq": format_filter.upper()}})

    if status_filter:
        conditions.append({"status": {"$eq": status_filter.upper()}})

    if franchise_root_id is not None:
        conditions.append({"franchise_root_id": {"$eq": franchise_root_id}})

    # Score range
    if min_score is not None:
        conditions.append({"average_score": {"$gte": min_score}})
    if max_score is not None:
        conditions.append({"average_score": {"$lte": max_score}})

    # Year range
    if min_year is not None:
        conditions.append({"season_year": {"$gte": min_year}})
    if max_year is not None:
        conditions.append({"season_year": {"$lte": max_year}})

    # Franchise exclusions (Step 30 deduplication)
    if exclude_franchise_root_ids:
        for root_id in exclude_franchise_root_ids:
            conditions.append({"franchise_root_id": {"$ne": root_id}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def parse_genres_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Recover genre list from the comma-joined ``genres`` metadata field.

    Args:
        metadata: Dict from ``build_vector_metadata()`` or ChromaDB query result.

    Returns:
        List of genre strings, or empty list if field is absent/blank.
    """
    raw = metadata.get("genres", "")
    return [g.strip() for g in raw.split(",") if g.strip()] if raw else []


def parse_top_tags_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Recover top tag list from the comma-joined ``top_tags`` metadata field.

    Args:
        metadata: Dict from ``build_vector_metadata()`` or ChromaDB query result.

    Returns:
        List of tag name strings, or empty list if field is absent/blank.
    """
    raw = metadata.get("top_tags", "")
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else []
