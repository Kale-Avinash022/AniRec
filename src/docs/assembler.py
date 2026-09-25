"""Document Text Formatter & Context Assembly (Step 17).

Role in the pipeline
--------------------
This is the final assembly stage of Phase 3.  It takes a ``SanitizedAnime``
(Step 12 output, enriched with ``franchise_root_id`` from Step 13) and
produces two things:

1. ``document_text`` — the embedding-ready string passed to the vector model
   (Step 20/23).
2. ``AnimeDocument`` — the complete ChromaDB record object (schema from
   Step 15), which bundles the document text with the typed metadata dict
   that powers ``where``-clause filtering (Step 18).

Document template
-----------------
::

    {Primary Title} ({Alternate Title}), {Year}, {Format}, {Episodes} episodes.
    Genres: {Genres}.
    Tags: {Weighted ranked tags}.
    Studio: {Studio}.
    Description: {Cleaned synopsis, capped at MAX_DESCRIPTION_CHARS}.
    Average score: {Score}/100. Popularity: {Popularity}.

Design choices
--------------
- **Title first**: Embedding models attend heavily to sequence starts; the
  show name anchors the semantic vector.
- **Tags before description**: High-signal weighted keywords prime the model
  before the long-form synopsis.
- **Description cap**: ``MAX_DESCRIPTION_CHARS`` from Step 15 keeps token
  budget predictable and prevents light-novel adaptations from dominating.
- **Graceful fallbacks**: Every field has a safe default so no show produces
  a broken or partial document.
- **Alternate title de-duplication**: When romaji and english are the same
  string (case-insensitive), only one is shown to avoid redundant tokens.
"""

from __future__ import annotations

from typing import Any

from src.docs.schema import (
    MAX_DESCRIPTION_CHARS,
    AnimeDocument,
)
from src.docs.tags import (
    compute_tag_density_score,
    format_weighted_tags,
    get_top_tag_names,
)
from src.ingest.transform import SanitizedAnime

# ---------------------------------------------------------------------------
# Core text assembler
# ---------------------------------------------------------------------------


def build_document_text(
    anime: SanitizedAnime,
    min_tag_rank: int = 40,
    max_tags: int = 15,
    max_description_chars: int = MAX_DESCRIPTION_CHARS,
) -> str:
    """Assemble the embedding-ready document string for a single anime.

    This is the primary entry point for Step 17.  The output is stored as
    ``AnimeDocument.document_text`` and passed directly to the embedding model
    in Step 23.

    Args:
        anime:                ``SanitizedAnime`` from Step 12.
        min_tag_rank:         Minimum community rank for tag inclusion.
        max_tags:             Maximum tags to include in the document.
        max_description_chars: Hard cap on synopsis length before truncation.

    Returns:
        Multi-line string ready for embedding — deterministic for the same
        input, always non-empty.
    """
    # ── Title line ──────────────────────────────────────────────────────────
    primary = anime.title_romaji.strip() or "Unknown Title"

    # Include alternate title only when it is meaningfully different
    alt = (anime.title_english or "").strip()
    if alt and alt.lower() != primary.lower():
        title_header = f"{primary} ({alt})"
    else:
        title_header = primary

    # ── Temporal / format fields ────────────────────────────────────────────
    year_str = str(anime.season_year) if anime.season_year else "Unknown Year"
    season_str = anime.season.capitalize() if anime.season else ""
    format_str = anime.format or "Unknown Format"

    # Episode count only makes sense for episodic formats
    non_episodic = {"MOVIE", "MUSIC", "MANGA", "NOVEL", "ONE_SHOT"}
    if anime.format in non_episodic or not anime.episodes:
        episodes_str = format_str
    else:
        episodes_str = f"{format_str}, {anime.episodes} episodes"

    # Combine into one compact header token
    season_label = f" {season_str}" if season_str else ""
    header = f"{title_header}, {year_str}{season_label}, {episodes_str}."

    # ── Taxonomy ────────────────────────────────────────────────────────────
    genres_str = ", ".join(anime.genres) if anime.genres else "General"
    tags_str = format_weighted_tags(anime.tags, min_rank=min_tag_rank, max_tags=max_tags)

    # ── Production ──────────────────────────────────────────────────────────
    studios_str = ", ".join(anime.studios) if anime.studios else "Unknown Studio"

    # ── Synopsis ────────────────────────────────────────────────────────────
    desc = anime.description_clean.strip()
    if len(desc) > max_description_chars:
        # Truncate at the last sentence boundary within the cap
        truncated = desc[:max_description_chars]
        last_period = truncated.rfind(".")
        if last_period > max_description_chars // 2:
            desc = truncated[: last_period + 1]
        else:
            desc = truncated.rstrip() + "..."
    description_str = desc if desc else "No description available."

    # ── Score / popularity ──────────────────────────────────────────────────
    score_str = f"{anime.average_score}/100" if anime.average_score is not None else "Unrated"
    popularity_str = str(anime.popularity) if anime.popularity else "Unknown"

    # ── Assemble ────────────────────────────────────────────────────────────
    document = (
        f"{header}\n"
        f"Genres: {genres_str}.\n"
        f"Tags: {tags_str}.\n"
        f"Studio: {studios_str}.\n"
        f"Description: {description_str}\n"
        f"Average score: {score_str}. Popularity: {popularity_str}."
    )
    return document.strip()


# ---------------------------------------------------------------------------
# Full AnimeDocument builder
# ---------------------------------------------------------------------------


def build_anime_document(
    anime: SanitizedAnime,
    franchise_root_id: int,
    min_tag_rank: int = 40,
    max_tags: int = 15,
) -> AnimeDocument:
    """Build a complete ``AnimeDocument`` from a sanitized anime + root ID.

    Combines the assembled document text (Step 17) with the typed metadata
    dict (Step 18 schema preview) into the canonical ``AnimeDocument``
    Pydantic object ready for ChromaDB upsert (Step 22/23).

    Args:
        anime:            ``SanitizedAnime`` from Step 12.
        franchise_root_id: From ``build_root_id_map()`` in Step 13.
        min_tag_rank:     Forwarded to ``build_document_text()``.
        max_tags:         Forwarded to ``build_document_text()``.

    Returns:
        Fully populated ``AnimeDocument`` instance.
    """
    document_text = build_document_text(
        anime,
        min_tag_rank=min_tag_rank,
        max_tags=max_tags,
    )

    # Step 18 will expand this — here we assemble the core scalar fields
    # that ChromaDB WHERE-clause filtering requires.
    # All values must be str, int, float, or bool (ChromaDB constraint).
    metadata: dict[str, Any] = {
        # Titles
        "title_romaji": anime.title_romaji,
        "title_english": anime.title_english or "",
        # Scores
        "average_score": anime.average_score if anime.average_score is not None else 0,
        "mean_score": anime.mean_score if anime.mean_score is not None else 0,
        "popularity": anime.popularity,
        "favourites": anime.favourites,
        # Temporal
        "season_year": anime.season_year if anime.season_year is not None else 0,
        "season": anime.season or "",
        # Classification
        "format": anime.format or "",
        "status": anime.status or "",
        "episodes": anime.episodes if anime.episodes is not None else 0,
        # Taxonomy (stored as comma-joined strings — ChromaDB doesn't support lists)
        "genres": ",".join(anime.genres) if anime.genres else "",
        "top_tags": ",".join(get_top_tag_names(anime.tags, n=5, min_rank=min_tag_rank)),
        # Production
        "studio": anime.studios[0] if anime.studios else "",
        # Quality signal
        "tag_density": compute_tag_density_score(anime.tags, min_rank=min_tag_rank),
        # Franchise
        "franchise_root_id": franchise_root_id,
        # Assets
        "cover_image_url": anime.cover_image_large or "",
        "site_url": anime.site_url or "",
        # Identifiers
        "anime_id": anime.id,
        "id_mal": anime.id_mal if anime.id_mal is not None else 0,
    }

    return AnimeDocument(
        doc_id=AnimeDocument.make_doc_id(anime.id),
        anime_id=anime.id,
        franchise_root_id=franchise_root_id,
        document_text=document_text,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def build_document_batch(
    catalog: list[SanitizedAnime],
    root_map: dict[int, int],
    min_tag_rank: int = 40,
    max_tags: int = 15,
) -> tuple[list[AnimeDocument], list[tuple[int, Exception]]]:
    """Build ``AnimeDocument`` objects for an entire sanitized catalog.

    Processes every item independently so a single assembly failure doesn't
    abort the batch.

    Args:
        catalog:      List of ``SanitizedAnime`` from Step 12.
        root_map:     ``{anime_id: franchise_root_id}`` from Step 13.
        min_tag_rank: Forwarded to each ``build_anime_document()`` call.
        max_tags:     Forwarded to each ``build_anime_document()`` call.

    Returns:
        2-tuple of:
        - ``documents``: Successfully assembled ``AnimeDocument`` list.
        - ``errors``:    ``[(index, exception), ...]`` for failed items.
    """
    documents: list[AnimeDocument] = []
    errors: list[tuple[int, Exception]] = []

    for idx, anime in enumerate(catalog):
        try:
            root_id = root_map.get(anime.id, anime.id)
            documents.append(
                build_anime_document(
                    anime,
                    franchise_root_id=root_id,
                    min_tag_rank=min_tag_rank,
                    max_tags=max_tags,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append((idx, exc))

    return documents, errors
