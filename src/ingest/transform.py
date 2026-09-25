"""Anime Metadata Extraction & Field Sanitization (Step 12).

Transforms raw AniList GraphQL media objects into strongly-typed
``SanitizedAnime`` domain models by:

- Stripping residual HTML tags and normalising whitespace in descriptions.
- Resolving multi-title representations (romaji / english / native).
- Filtering spoiler-tagged community keywords.
- Sorting tags by rank (descending) for weighted downstream embedding.
- Extracting primary studio names from nested node lists.
- Coalescing null scores, years, and episode counts to typed-safe defaults.
- Flattening relation edges into a concise edge list for graph modelling.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


class SanitizedTag(BaseModel):
    """A single community tag attached to an anime, stripped of spoiler flags."""

    id: int
    name: str
    rank: int  # 0-100 weight used for downstream TF-IDF-style blending
    is_spoiler: bool = False


class RelationEdge(BaseModel):
    """A directed franchise relationship edge from one media entry to another."""

    relation_type: str | None = None  # e.g. SEQUEL, PREQUEL, SIDE_STORY
    target_id: int | None = None
    target_title_romaji: str | None = None
    target_title_english: str | None = None
    target_format: str | None = None  # e.g. TV, MOVIE, OVA


class SanitizedAnime(BaseModel):
    """Cleaned, normalised domain object produced from a raw AniList media node."""

    # --- Identifiers ---
    id: int
    id_mal: int | None = None  # MyAnimeList mirror ID

    # --- Titles ---
    title_romaji: str
    title_english: str | None = None
    title_native: str | None = None

    # --- Content ---
    description_clean: str = ""

    # --- Taxonomy ---
    genres: list[str] = Field(default_factory=list)
    tags: list[SanitizedTag] = Field(default_factory=list)

    # --- Scores & Popularity ---
    average_score: int | None = None   # AniList community weighted score (0-100)
    mean_score: int | None = None      # Raw mean vote (0-100)
    popularity: int = 0
    favourites: int = 0

    # --- Temporal ---
    season_year: int | None = None
    season: str | None = None          # WINTER | SPRING | SUMMER | FALL

    # --- Format & Runtime ---
    format: str | None = None          # TV | MOVIE | OVA | ONA | SPECIAL | MUSIC
    episodes: int | None = None
    duration: int | None = None        # minutes per episode

    # --- Status ---
    status: str | None = None          # FINISHED | RELEASING | NOT_YET_RELEASED | CANCELLED

    # --- Production ---
    studios: list[str] = Field(default_factory=list)

    # --- Media Assets ---
    cover_image_large: str | None = None
    cover_image_extra_large: str | None = None
    banner_image: str | None = None
    site_url: str | None = None

    # --- Franchise Graph ---
    relations: list[RelationEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sanitization Helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_html_description(raw_text: str | None) -> str:
    """Strip residual HTML tags and collapse whitespace in a description string.

    AniList returns ``description(asHtml: false)`` which is mostly clean, but
    some entries still contain ``<br>``, ``<i>``, ``<b>``, and entity-encoded
    characters.  This function provides a robust safety net.

    Args:
        raw_text: Raw description string, potentially containing HTML.

    Returns:
        Plain-text description with normalised whitespace, or ``""`` if blank.
    """
    if not raw_text:
        return ""
    text = _HTML_TAG_RE.sub(" ", raw_text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _extract_tags(raw_tags: list[dict[str, Any]]) -> list[SanitizedTag]:
    """Parse and sort non-spoiler tags by rank descending.

    Args:
        raw_tags: Raw tag list from AniList ``tags`` field.

    Returns:
        Sorted list of ``SanitizedTag`` objects (highest rank first).
    """
    sanitized: list[SanitizedTag] = []
    for t in raw_tags:
        if not t:
            continue
        if t.get("isMediaSpoiler", False):
            continue  # Exclude spoiler-tagged keywords from public embeddings
        sanitized.append(
            SanitizedTag(
                id=t.get("id") or 0,
                name=t.get("name") or "",
                rank=t.get("rank") or 0,
                is_spoiler=False,
            )
        )
    sanitized.sort(key=lambda x: x.rank, reverse=True)
    return sanitized


def _extract_studios(raw_studios: dict[str, Any]) -> list[str]:
    """Extract primary studio names from the nested studios node structure.

    Args:
        raw_studios: Value of ``studios`` key in a raw media object.

    Returns:
        List of studio name strings (may be empty).
    """
    nodes: list[dict[str, Any]] = (raw_studios or {}).get("nodes") or []
    return [s["name"] for s in nodes if s and s.get("name")]


def _extract_relations(raw_relations: dict[str, Any]) -> list[RelationEdge]:
    """Flatten relation edge list from the nested relations structure.

    Args:
        raw_relations: Value of ``relations`` key in a raw media object.

    Returns:
        List of ``RelationEdge`` objects representing franchise links.
    """
    edges: list[dict[str, Any]] = (raw_relations or {}).get("edges") or []
    result: list[RelationEdge] = []
    for edge in edges:
        if not edge:
            continue
        node: dict[str, Any] = edge.get("node") or {}
        if not node:
            continue
        node_titles: dict[str, Any] = node.get("title") or {}
        result.append(
            RelationEdge(
                relation_type=edge.get("relationType"),
                target_id=node.get("id"),
                target_title_romaji=node_titles.get("romaji"),
                target_title_english=node_titles.get("english"),
                target_format=node.get("format"),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public Transformation Entry Point
# ---------------------------------------------------------------------------


def sanitize_raw_media(raw: dict[str, Any]) -> SanitizedAnime:
    """Transform a single raw AniList GraphQL media node into a ``SanitizedAnime``.

    This is the main entry point for Step 12.  It is called once per media item
    emitted by ``AniListClient.ingest_catalog()``.

    Args:
        raw: Raw media dictionary as returned by the AniList GraphQL API.

    Returns:
        A fully validated ``SanitizedAnime`` Pydantic model instance.

    Raises:
        ValueError: If the raw dict does not contain a valid integer ``id``.
    """
    media_id = raw.get("id")
    if not isinstance(media_id, int):
        raise ValueError(f"Raw media object missing valid 'id' field: {raw!r:.120}")

    titles: dict[str, Any] = raw.get("title") or {}
    cover: dict[str, Any] = raw.get("coverImage") or {}

    return SanitizedAnime(
        # Identifiers
        id=media_id,
        id_mal=raw.get("idMal"),
        # Titles
        title_romaji=titles.get("romaji") or "Unknown Title",
        title_english=titles.get("english") or None,
        title_native=titles.get("native") or None,
        # Content
        description_clean=clean_html_description(raw.get("description")),
        # Taxonomy
        genres=raw.get("genres") or [],
        tags=_extract_tags(raw.get("tags") or []),
        # Scores
        average_score=raw.get("averageScore"),
        mean_score=raw.get("meanScore"),
        popularity=raw.get("popularity") or 0,
        favourites=raw.get("favourites") or 0,
        # Temporal
        season_year=raw.get("seasonYear"),
        season=raw.get("season"),
        # Format
        format=raw.get("format"),
        episodes=raw.get("episodes"),
        duration=raw.get("duration"),
        # Status
        status=raw.get("status"),
        # Production
        studios=_extract_studios(raw.get("studios") or {}),
        # Assets
        cover_image_large=cover.get("large"),
        cover_image_extra_large=cover.get("extraLarge"),
        banner_image=raw.get("bannerImage"),
        site_url=raw.get("siteUrl"),
        # Franchise graph
        relations=_extract_relations(raw.get("relations") or {}),
    )


def sanitize_raw_media_batch(
    raw_list: list[dict[str, Any]],
) -> tuple[list[SanitizedAnime], list[tuple[int, Exception]]]:
    """Transform a batch of raw media objects, collecting per-item errors.

    Processes every item independently so a single malformed record does not
    abort the entire ingestion batch.

    Args:
        raw_list: List of raw media dicts from ``AniListClient.ingest_catalog()``.

    Returns:
        A 2-tuple of:
        - ``results``: Successfully sanitized ``SanitizedAnime`` instances.
        - ``errors``: List of ``(index, exception)`` pairs for failed items.
    """
    results: list[SanitizedAnime] = []
    errors: list[tuple[int, Exception]] = []

    for idx, raw in enumerate(raw_list):
        try:
            results.append(sanitize_raw_media(raw))
        except Exception as exc:  # noqa: BLE001
            errors.append((idx, exc))

    return results, errors
