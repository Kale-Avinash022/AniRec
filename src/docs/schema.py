"""Single-Document-Per-Show Paradigm — Schema & Design Constants (Step 15).

Architecture Decision: Why One Document Per Show?
-------------------------------------------------
Traditional RAG pipelines split source material into fixed-size chunks
(typically 300–500 tokens) because they work with long-form unstructured
text (PDFs, wiki articles) where different passages answer different
questions.

Anime recommendation is fundamentally different.  Each query is an
**entity-level judgment** ("find me shows like X that have property Y"),
not a passage retrieval task.  The answer is a *whole show*, not a
fragment of one.

Chunking a show's synopsis across boundaries creates three hard failures:

1. **Identity destruction** — A vector for the "action" chunk of *Attack
   on Titan* has no awareness it belongs to a show with score 90, format TV,
   year 2013, studio WIT.  Metadata lives in a different chunk.

2. **Tag-genre disconnection** — Community tags (e.g. "Dark Fantasy (95)",
   "Military (88)") are stored separately from the synopsis chunk that
   references them.  The embedding misses the thematic fingerprint.

3. **Franchise confusion** — A sequel's synopsis might reference characters
   by name from the original; chunking makes it impossible to know which
   franchise entry retrieved this passage without expensive re-attribution.

The Single-Document-Per-Show design avoids all three failures by engineering
one rich, structured text document per anime entity.  The document surface
combines titles, genres, weighted tags, synopsis, and studio into a single
embedding-ready string so that one vector captures the show's complete
semantic fingerprint.

This schema is the contract that Steps 16 (tag weighting), 17 (text assembly),
18 (metadata schema), 19 (persistence), and 20+ (embedding + indexing) all
implement against.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Design constants (referenced by Steps 16, 17, 22, 27)
# ---------------------------------------------------------------------------

CHUNKING_STRATEGY: str = "single_doc_per_show"
"""The chosen embedding strategy identifier.

Stored as a ChromaDB collection-level metadata tag so retrieval code can
assert it is operating against a correctly structured index.
"""

MAX_TAG_SLOTS: int = 15
"""Maximum number of weighted tags included in the synthesized document.

Tags are sorted by rank descending (Step 16).  Beyond 15 the marginal
semantic signal is negligible and token budget grows without benefit.
"""

MAX_DESCRIPTION_CHARS: int = 1_200
"""Hard cap on description length before truncation.

AniList descriptions occasionally exceed 3,000 characters (detailed light
novel adaptations).  Capping at ~1,200 keeps token usage predictable for
both the embedding model (Step 20) and the LLM generation layer (Step 33).
"""

DOCUMENT_VERSION: str = "1.0"
"""Schema version embedded in processed snapshots (Step 19).

Increment this when the document format changes to enable cache invalidation.
"""

FRANCHISE_RELATION_LABEL: str = "franchise_root_id"
"""The metadata key name used throughout Steps 18, 22, 27, 30.

Centralised here so a single rename propagates everywhere.
"""


# ---------------------------------------------------------------------------
# AnimeDocument — canonical document type for Phase 3 and beyond
# ---------------------------------------------------------------------------


class AnimeDocument(BaseModel):
    """The canonical unit of storage and retrieval in AniRec.

    An ``AnimeDocument`` is produced by Steps 16–17 and consumed by:
    - Step 18 (metadata schema)
    - Step 19 (processed snapshot persistence)
    - Step 22 (ChromaDB upsert)
    - Steps 25–31 (retrieval pipeline)
    - Steps 32–37 (generation / grounding)

    Fields
    ------
    doc_id          : Stable string key for ChromaDB — always ``"anilist_{id}"``.
    anime_id        : Raw AniList integer ID.
    franchise_root_id: Lowest ID in this show's franchise cluster (Step 13).
    document_text   : The embedding-ready synthesized string (Step 17 output).
    metadata        : Flat dict of typed scalar fields for ChromaDB ``where``
                      filtering (Step 18 schema).
    schema_version  : Matches ``DOCUMENT_VERSION`` at build time.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    doc_id: str = Field(
        ...,
        description="ChromaDB document key — format: 'anilist_{anime_id}'",
        examples=["anilist_16498"],
    )
    anime_id: int = Field(..., description="AniList integer media ID.")
    franchise_root_id: int = Field(
        ...,
        description="Minimum ID in the franchise cluster (Step 13 output).",
    )

    # ── Embedding surface ───────────────────────────────────────────────────
    document_text: str = Field(
        ...,
        description=(
            "The synthesized, embedding-ready text document for this show. "
            "Produced by Step 17 assembler."
        ),
    )

    # ── Metadata (Step 18 schema) ───────────────────────────────────────────
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flat dict of typed scalar fields passed to ChromaDB as metadata. "
            "Values must be str, int, float, or bool — no nested objects."
        ),
    )

    # ── Provenance ──────────────────────────────────────────────────────────
    schema_version: str = Field(
        default=DOCUMENT_VERSION,
        description="Document schema version for cache invalidation.",
    )

    # ── Convenience properties ──────────────────────────────────────────────

    @property
    def title(self) -> str:
        """Best available display title pulled from metadata."""
        return (
            self.metadata.get("title_english")
            or self.metadata.get("title_romaji")
            or f"Anime #{self.anime_id}"
        )

    @property
    def average_score(self) -> int | None:
        """Community score (0–100) from metadata, or None."""
        v = self.metadata.get("average_score")
        return int(v) if v is not None else None

    @property
    def season_year(self) -> int | None:
        """Release year from metadata, or None."""
        v = self.metadata.get("season_year")
        return int(v) if v is not None else None

    @classmethod
    def make_doc_id(cls, anime_id: int) -> str:
        """Generate the canonical ChromaDB document ID for an anime.

        Args:
            anime_id: AniList integer media ID.

        Returns:
            Stable string key in the format ``'anilist_{anime_id}'``.
        """
        return f"anilist_{anime_id}"


# ---------------------------------------------------------------------------
# Design justification table (machine-readable for docs / tests)
# ---------------------------------------------------------------------------

CHUNKING_STRATEGY_COMPARISON: list[dict[str, str]] = [
    {
        "approach": "Fixed Token Chunking (500 tokens)",
        "mechanism": "Splits synopsis across chunk boundaries; isolates tags from genres",
        "consequence": "A query matching a plot chunk loses title, year, format, and score metadata",
        "verdict": "REJECTED — destroys holistic entity identity",
    },
    {
        "approach": "Sentence-Level Chunking",
        "mechanism": "Each synopsis sentence becomes an individual vector",
        "consequence": "Multi-sentence thematic arcs fragmented; scores and tags lost",
        "verdict": "REJECTED — high noise, loss of show context",
    },
    {
        "approach": "Curated Whole-Show Document Synthesis",
        "mechanism": "One rich document per show: Title + Year + Genres + Weighted Tags + Synopsis + Studio",
        "consequence": "Vector represents full thematic fingerprint with atomic metadata retained",
        "verdict": "SELECTED — preserves entity integrity",
    },
]
