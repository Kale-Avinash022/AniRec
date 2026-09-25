"""Processed Snapshot Persistence — data/processed (Step 19).

What this module does
---------------------
After the full Phase 2 + Phase 3 pipeline has run (Steps 12–18), every anime
has been transformed into an ``AnimeDocument`` with:

  - A synthesized embedding-ready ``document_text``   (Step 17)
  - A type-safe flat ``metadata`` dict                (Step 18)
  - A ``franchise_root_id``                           (Step 13)

This module persists that enriched corpus to ``data/processed/`` as JSON so
that downstream stages (Step 22 ChromaDB indexing, evaluation, debugging) can
reload it without re-running the expensive AniList ingestion and transformation
pipeline.

data/ directory layout after this step
---------------------------------------
::

    data/
    ├── raw/              ← Step 10 (page_001.json … page_060.json)
    └── processed/
        ├── anime_corpus.json      ← main corpus (list of serialized AnimeDocuments)
        └── corpus_manifest.json   ← metadata about the snapshot (count, version, timestamp)

Design decisions
----------------
- **Atomic writes**: Write to ``.tmp`` first, then ``os.replace()`` so a crash
  mid-write never leaves a corrupt file.
- **Versioned manifest**: ``corpus_manifest.json`` records the schema version,
  record count, and UTC timestamp — supports cache invalidation when the
  document format changes (``DOCUMENT_VERSION`` in Step 15).
- **Round-trip fidelity**: ``AnimeDocument`` Pydantic models serialize via
  ``.model_dump()`` and reload via ``AnimeDocument.model_validate()``.
- **Incremental ID lookup**: ``load_document_by_id()`` scans the corpus for a
  single doc without loading the full file — used by retrieval debug tools.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logger import logger
from src.docs.schema import DOCUMENT_VERSION, AnimeDocument

# ---------------------------------------------------------------------------
# File name constants
# ---------------------------------------------------------------------------

CORPUS_FILENAME: str = "anime_corpus.json"
MANIFEST_FILENAME: str = "corpus_manifest.json"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _build_manifest(
    record_count: int,
    output_path: Path,
) -> dict[str, Any]:
    """Build a corpus manifest dict for audit and cache-invalidation purposes."""
    return {
        "schema_version": DOCUMENT_VERSION,
        "record_count": record_count,
        "corpus_file": output_path.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _save_manifest(manifest: dict[str, Any], processed_dir: Path) -> Path:
    manifest_path = processed_dir / MANIFEST_FILENAME
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, manifest_path)
    return manifest_path


def load_manifest(processed_dir: Path | None = None) -> dict[str, Any] | None:
    """Load the corpus manifest if it exists.

    Args:
        processed_dir: Override directory; defaults to ``settings.PROCESSED_DATA_DIR``.

    Returns:
        Manifest dict, or ``None`` if no manifest exists yet.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    path = target_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_processed_corpus(
    documents: list[AnimeDocument],
    filename: str = CORPUS_FILENAME,
    processed_dir: Path | None = None,
) -> Path:
    """Atomically persist a list of ``AnimeDocument`` objects to disk.

    Serialises each document via ``model_dump()`` and writes to a ``.tmp``
    file before an atomic rename, guaranteeing file integrity even if the
    process is killed mid-write.

    Also writes a ``corpus_manifest.json`` alongside the corpus for
    cache-invalidation and audit purposes.

    Args:
        documents:     List of fully assembled ``AnimeDocument`` instances.
        filename:      Output filename (default ``anime_corpus.json``).
        processed_dir: Override directory; defaults to ``settings.PROCESSED_DATA_DIR``.

    Returns:
        Resolved ``Path`` of the written corpus file.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    output_path = target_dir / filename
    tmp_path = output_path.with_suffix(".tmp")

    # Serialise all documents
    serialised = [doc.model_dump() for doc in documents]

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)

    # Atomic rename
    os.replace(tmp_path, output_path)

    # Write manifest
    manifest = _build_manifest(len(documents), output_path)
    manifest_path = _save_manifest(manifest, target_dir)

    logger.info(
        f"Saved {len(documents)} processed AnimeDocuments to {output_path} "
        f"| manifest: {manifest_path.name} "
        f"| schema_version: {DOCUMENT_VERSION}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_processed_corpus(
    filename: str = CORPUS_FILENAME,
    processed_dir: Path | None = None,
) -> list[AnimeDocument]:
    """Load the full processed corpus from disk as validated ``AnimeDocument`` objects.

    Args:
        filename:      Corpus filename to load (default ``anime_corpus.json``).
        processed_dir: Override directory; defaults to ``settings.PROCESSED_DATA_DIR``.

    Returns:
        List of ``AnimeDocument`` instances.

    Raises:
        FileNotFoundError: If the corpus file does not exist.
        ValueError:        If the corpus file is corrupt or schema-incompatible.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    path = target_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found at {path}. "
            "Run the ingestion pipeline first (Steps 12–18)."
        )

    with open(path, encoding="utf-8") as f:
        raw_list: list[dict[str, Any]] = json.load(f)

    documents: list[AnimeDocument] = []
    errors: list[tuple[int, Exception]] = []

    for idx, raw in enumerate(raw_list):
        try:
            documents.append(AnimeDocument.model_validate(raw))
        except Exception as exc:  # noqa: BLE001
            errors.append((idx, exc))

    if errors:
        logger.warning(
            f"Loaded {len(documents)} documents with {len(errors)} validation errors "
            f"(first error at index {errors[0][0]}: {errors[0][1]})"
        )
    else:
        logger.info(f"Loaded {len(documents)} AnimeDocuments from {path}")

    return documents


def load_corpus_as_dicts(
    filename: str = CORPUS_FILENAME,
    processed_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load the corpus as raw dicts (no Pydantic validation).

    Faster than ``load_processed_corpus()`` when you only need dict-level
    access (e.g. bulk ChromaDB upsert in Step 23).

    Args:
        filename:      Corpus filename (default ``anime_corpus.json``).
        processed_dir: Override directory.

    Returns:
        List of raw dicts as stored in the JSON file.

    Raises:
        FileNotFoundError: If the corpus file does not exist.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    path = target_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Processed corpus not found at {path}. "
            "Run the ingestion pipeline first."
        )

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_document_by_id(
    anime_id: int,
    filename: str = CORPUS_FILENAME,
    processed_dir: Path | None = None,
) -> AnimeDocument | None:
    """Find and return a single document by AniList ID without loading the full corpus.

    Intended for debugging and retrieval validation tools.  For bulk access,
    use ``load_processed_corpus()`` and build an in-memory index.

    Args:
        anime_id:      AniList integer ID to search for.
        filename:      Corpus filename.
        processed_dir: Override directory.

    Returns:
        Matching ``AnimeDocument`` or ``None`` if not found.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    path = target_dir / filename

    if not path.exists():
        return None

    with open(path, encoding="utf-8") as f:
        raw_list: list[dict[str, Any]] = json.load(f)

    target_doc_id = AnimeDocument.make_doc_id(anime_id)
    for raw in raw_list:
        if raw.get("doc_id") == target_doc_id or raw.get("anime_id") == anime_id:
            try:
                return AnimeDocument.model_validate(raw)
            except Exception:  # noqa: BLE001
                return None
    return None


def corpus_exists(
    filename: str = CORPUS_FILENAME,
    processed_dir: Path | None = None,
) -> bool:
    """Check whether a saved corpus exists on disk.

    Used by the ingestion pipeline to decide whether to skip re-processing.

    Args:
        filename:      Corpus filename.
        processed_dir: Override directory.

    Returns:
        ``True`` if the corpus file exists on disk.
    """
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR
    return (target_dir / filename).exists()


def get_corpus_stats(
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Return quick stats about the saved corpus without loading all documents.

    Reads the manifest only — O(1) regardless of corpus size.

    Args:
        processed_dir: Override directory.

    Returns:
        Dict with ``record_count``, ``schema_version``, ``created_at_utc``,
        and ``corpus_exists`` keys.
    """
    manifest = load_manifest(processed_dir)
    target_dir = processed_dir or settings.PROCESSED_DATA_DIR

    return {
        "corpus_exists": corpus_exists(processed_dir=target_dir),
        "record_count": manifest.get("record_count", 0) if manifest else 0,
        "schema_version": manifest.get("schema_version", "unknown") if manifest else "unknown",
        "created_at_utc": manifest.get("created_at_utc", None) if manifest else None,
    }
