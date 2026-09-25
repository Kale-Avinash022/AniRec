"""Document Engineering module (Phase 3: Steps 15–19).

This package implements the Single-Document-Per-Show paradigm decided in
Step 15 and realised in Steps 16–19.

Modules
-------
- schema    : AnimeDocument — the canonical document type for Phase 3+
- tags      : Weighted tag rank integration engine (Step 16)
- assembler : Document text formatter & context assembly (Step 17)
- metadata  : Metadata schema construction for vector filtering (Step 18)
- snapshot  : Processed snapshot persistence to data/processed (Step 19)
"""

from src.docs.assembler import (
    build_anime_document,
    build_document_batch,
    build_document_text,
)
from src.docs.metadata import (
    EQUALITY_FILTER_FIELDS,
    FIELD_OPERATORS,
    METADATA_FIELD_TYPES,
    RANGE_FILTER_FIELDS,
    build_vector_metadata,
    build_where_clause,
    parse_genres_from_metadata,
    parse_top_tags_from_metadata,
    validate_metadata,
)
from src.docs.schema import (
    CHUNKING_STRATEGY,
    CHUNKING_STRATEGY_COMPARISON,
    DOCUMENT_VERSION,
    FRANCHISE_RELATION_LABEL,
    MAX_DESCRIPTION_CHARS,
    MAX_TAG_SLOTS,
    AnimeDocument,
)
from src.docs.snapshot import (
    CORPUS_FILENAME,
    MANIFEST_FILENAME,
    corpus_exists,
    get_corpus_stats,
    load_corpus_as_dicts,
    load_document_by_id,
    load_manifest,
    load_processed_corpus,
    save_processed_corpus,
)
from src.docs.tags import (
    DEFAULT_MAX_TAGS,
    DEFAULT_MIN_RANK,
    HIGH_CONFIDENCE_THRESHOLD,
    compute_tag_density_score,
    format_weighted_tags,
    get_top_tag_names,
    split_tags_by_confidence,
)

__all__ = [
    # Step 15 — Schema
    "CHUNKING_STRATEGY",
    "CHUNKING_STRATEGY_COMPARISON",
    "DOCUMENT_VERSION",
    "FRANCHISE_RELATION_LABEL",
    "MAX_DESCRIPTION_CHARS",
    "MAX_TAG_SLOTS",
    "AnimeDocument",
    # Step 16 — Tags
    "DEFAULT_MAX_TAGS",
    "DEFAULT_MIN_RANK",
    "HIGH_CONFIDENCE_THRESHOLD",
    "compute_tag_density_score",
    "format_weighted_tags",
    "get_top_tag_names",
    "split_tags_by_confidence",
    # Step 17 — Assembler
    "build_anime_document",
    "build_document_batch",
    "build_document_text",
    # Step 18 — Metadata
    "EQUALITY_FILTER_FIELDS",
    "FIELD_OPERATORS",
    "METADATA_FIELD_TYPES",
    "RANGE_FILTER_FIELDS",
    "build_vector_metadata",
    "build_where_clause",
    "parse_genres_from_metadata",
    "parse_top_tags_from_metadata",
    "validate_metadata",
    # Step 19 — Snapshot
    "CORPUS_FILENAME",
    "MANIFEST_FILENAME",
    "corpus_exists",
    "get_corpus_stats",
    "load_corpus_as_dicts",
    "load_document_by_id",
    "load_manifest",
    "load_processed_corpus",
    "save_processed_corpus",
]
