"""Data Validation & Quality Pipeline with Pydantic (Step 14).

This module is the **final stage of Phase 2 (Ingestion)**.  It ties together
the outputs of Steps 12 and 13 and performs a structured audit across the
entire sanitized catalog before any data is committed to disk or indexed into
ChromaDB.

Pipeline contract
-----------------
::

    raw_pages (Step 11)
        └─► sanitize_raw_media_batch()   (Step 12)  →  SanitizedAnime list
                └─► build_root_id_map()  (Step 13)  →  franchise_root_id per item
                        └─► DataQualityAuditor       (Step 14)  →  AuditReport
                                └─► inject franchise_root_id into each record
                                        └─► validated catalog ready for Phase 3

Quality checks performed
------------------------
1. **ID uniqueness** — duplicate AniList IDs surface ingestion bugs.
2. **Title presence** — at least one of romaji / english must be non-empty.
3. **Content presence** — description OR ≥1 tag must exist for embedding.
4. **Score range**    — average_score must be in [0, 100] if present.
5. **Year plausibility** — season_year in [1960, 2030] if present.
6. **Franchise root validity** — every record's franchise_root_id must point
   to a registered ID (guards against phantom relation targets).

The auditor never raises on violations — it collects them into a structured
``AuditReport`` and logs a summary.  The caller decides whether to hard-fail
or proceed with warnings (configurable via ``raise_on_critical``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.core.logger import logger
from src.ingest.franchise_graph import build_root_id_map
from src.ingest.transform import SanitizedAnime, sanitize_raw_media_batch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_YEAR = 1960
MAX_YEAR = 2030


# ---------------------------------------------------------------------------
# AuditReport — structured result object
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AuditReport:
    """Structured output of a ``DataQualityAuditor.run_audit()`` call.

    All ``*_ids`` fields contain the AniList IDs of affected records so callers
    can inspect or quarantine individual entries rather than just seeing counts.
    """

    total_records: int
    unique_records: int

    # Check 1 — ID uniqueness
    duplicate_ids: list[int] = dataclasses.field(default_factory=list)

    # Check 2 — Title presence
    missing_title_ids: list[int] = dataclasses.field(default_factory=list)

    # Check 3 — Content presence (description + tags)
    empty_content_ids: list[int] = dataclasses.field(default_factory=list)

    # Check 4 — Score range
    invalid_score_ids: list[int] = dataclasses.field(default_factory=list)

    # Check 5 — Year plausibility
    implausible_year_ids: list[int] = dataclasses.field(default_factory=list)

    # Check 6 — Franchise root validity
    orphan_root_ids: list[int] = dataclasses.field(default_factory=list)

    # ── Derived metrics ────────────────────────────────────────────────────

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_ids)

    @property
    def missing_title_count(self) -> int:
        return len(self.missing_title_ids)

    @property
    def empty_content_count(self) -> int:
        return len(self.empty_content_ids)

    @property
    def invalid_score_count(self) -> int:
        return len(self.invalid_score_ids)

    @property
    def implausible_year_count(self) -> int:
        return len(self.implausible_year_ids)

    @property
    def orphan_root_count(self) -> int:
        return len(self.orphan_root_ids)

    @property
    def total_violations(self) -> int:
        """Sum of all individual check violation counts."""
        return (
            self.duplicate_count
            + self.missing_title_count
            + self.empty_content_count
            + self.invalid_score_count
            + self.implausible_year_count
            + self.orphan_root_count
        )

    @property
    def data_health_score(self) -> float:
        """0–100 composite health score.

        Deducts points for missing titles and empty content (the two issues
        most likely to produce silent embedding failures).  Other violations
        lower the score but are weighted less critically.
        """
        if self.total_records == 0:
            return 0.0
        critical = self.missing_title_count + self.empty_content_count
        minor = (
            self.duplicate_count
            + self.invalid_score_count
            + self.implausible_year_count
            + self.orphan_root_count
        )
        penalty = (critical * 2.0 + minor * 0.5) / self.total_records
        return round(max(0.0, (1.0 - penalty) * 100), 2)

    @property
    def is_clean(self) -> bool:
        """True only when zero violations were found across all checks."""
        return self.total_violations == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON logging / storage."""
        return {
            "total_records": self.total_records,
            "unique_records": self.unique_records,
            "duplicate_count": self.duplicate_count,
            "missing_title_count": self.missing_title_count,
            "empty_content_count": self.empty_content_count,
            "invalid_score_count": self.invalid_score_count,
            "implausible_year_count": self.implausible_year_count,
            "orphan_root_count": self.orphan_root_count,
            "total_violations": self.total_violations,
            "data_health_score": self.data_health_score,
            "is_clean": self.is_clean,
        }

    def __str__(self) -> str:
        lines = [
            "=== Data Quality Audit Report ===",
            f"  Total records   : {self.total_records}",
            f"  Unique records  : {self.unique_records}",
            f"  Duplicates      : {self.duplicate_count}",
            f"  Missing titles  : {self.missing_title_count}",
            f"  Empty content   : {self.empty_content_count}",
            f"  Invalid scores  : {self.invalid_score_count}",
            f"  Bad years       : {self.implausible_year_count}",
            f"  Orphan roots    : {self.orphan_root_count}",
            f"  Total violations: {self.total_violations}",
            f"  Health score    : {self.data_health_score}%",
            f"  Status          : {'CLEAN' if self.is_clean else 'VIOLATIONS FOUND'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataQualityAuditor
# ---------------------------------------------------------------------------


class DataQualityAuditor:
    """Runs structured quality checks against a sanitized anime catalog.

    Args:
        catalog:   List of ``SanitizedAnime`` from Step 12.
        root_map:  ``{anime_id: franchise_root_id}`` mapping from Step 13.
                   Pass ``None`` to skip franchise root validation.
    """

    def __init__(
        self,
        catalog: list[SanitizedAnime],
        root_map: dict[int, int] | None = None,
    ) -> None:
        self._catalog = catalog
        self._root_map = root_map
        self._all_ids: set[int] = {a.id for a in catalog}

    def run_audit(self, raise_on_critical: bool = False) -> AuditReport:
        """Execute all quality checks and return a structured ``AuditReport``.

        Args:
            raise_on_critical: If ``True``, raises ``ValueError`` when
                               duplicate IDs or missing titles are found.

        Returns:
            ``AuditReport`` instance with full violation details.
        """
        seen_ids: set[int] = set()
        duplicate_ids: list[int] = []
        missing_title_ids: list[int] = []
        empty_content_ids: list[int] = []
        invalid_score_ids: list[int] = []
        implausible_year_ids: list[int] = []
        orphan_root_ids: list[int] = []

        for anime in self._catalog:
            aid = anime.id

            # ── Check 1: Unique IDs ────────────────────────────────────────
            if aid in seen_ids:
                duplicate_ids.append(aid)
            seen_ids.add(aid)

            # ── Check 2: Title presence ────────────────────────────────────
            if not (anime.title_romaji.strip() or (anime.title_english or "").strip()):
                missing_title_ids.append(aid)

            # ── Check 3: Content presence ──────────────────────────────────
            has_description = bool(anime.description_clean.strip())
            has_tags = len(anime.tags) > 0
            if not has_description and not has_tags:
                empty_content_ids.append(aid)

            # ── Check 4: Score range ───────────────────────────────────────
            if anime.average_score is not None and not (
                0 <= anime.average_score <= 100
            ):
                invalid_score_ids.append(aid)

            # ── Check 5: Year plausibility ─────────────────────────────────
            if anime.season_year is not None and not (
                MIN_YEAR <= anime.season_year <= MAX_YEAR
            ):
                implausible_year_ids.append(aid)

            # ── Check 6: Franchise root validity ───────────────────────────
            if self._root_map is not None:
                root = self._root_map.get(aid)
                if root is not None and root not in self._all_ids and root != aid:
                    orphan_root_ids.append(aid)

        report = AuditReport(
            total_records=len(self._catalog),
            unique_records=len(seen_ids),
            duplicate_ids=duplicate_ids,
            missing_title_ids=missing_title_ids,
            empty_content_ids=empty_content_ids,
            invalid_score_ids=invalid_score_ids,
            implausible_year_ids=implausible_year_ids,
            orphan_root_ids=orphan_root_ids,
        )

        logger.info(
            f"Data Quality Audit complete — "
            f"Health: {report.data_health_score}% | "
            f"Violations: {report.total_violations}"
        )

        if raise_on_critical and (report.duplicate_count or report.missing_title_count):
            raise ValueError(
                f"Critical data quality violations found: "
                f"{report.duplicate_count} duplicate IDs, "
                f"{report.missing_title_count} missing titles."
            )

        return report


# ---------------------------------------------------------------------------
# franchise_root_id injection
# ---------------------------------------------------------------------------


def inject_franchise_roots(
    catalog: list[SanitizedAnime],
    root_map: dict[int, int],
) -> list[dict[str, Any]]:
    """Merge ``franchise_root_id`` from Step 13 into the catalog records.

    ``SanitizedAnime`` is a Pydantic model without a ``franchise_root_id``
    field — that field lives in the Step 18 vector-filter metadata schema.
    This function serialises each model to a dict and injects the root ID,
    producing the enriched records that Phase 3 (document assembly) will
    consume.

    Args:
        catalog:  Validated ``SanitizedAnime`` list.
        root_map: ``{anime_id: franchise_root_id}`` from ``build_root_id_map()``.

    Returns:
        List of enriched dicts, one per anime, with ``franchise_root_id`` key.
    """
    enriched: list[dict[str, Any]] = []
    for anime in catalog:
        record = anime.model_dump()
        record["franchise_root_id"] = root_map.get(anime.id, anime.id)
        enriched.append(record)
    return enriched


# ---------------------------------------------------------------------------
# Full Phase 2 pipeline runner
# ---------------------------------------------------------------------------


def run_full_pipeline(
    raw_pages: list[list[dict[str, Any]]],
    raise_on_critical: bool = False,
) -> tuple[list[dict[str, Any]], AuditReport]:
    """Execute the complete Phase 2 ingestion pipeline in one call.

    Chains Steps 12 → 13 → 14 together:
    1. Flatten raw page snapshots into a single media list.
    2. Sanitize via ``sanitize_raw_media_batch()``   (Step 12).
    3. Build franchise root map via ``build_root_id_map()`` (Step 13).
    4. Run quality audit via ``DataQualityAuditor``  (Step 14).
    5. Inject ``franchise_root_id`` into each record.
    6. Return enriched records + audit report.

    Args:
        raw_pages:        List of raw page dicts from ``AniListClient.ingest_catalog()``.
                          Can be either the nested ``[page_response, ...]`` format or
                          a flat ``[media_dict, ...]`` list.
        raise_on_critical: Forward to ``DataQualityAuditor.run_audit()``.

    Returns:
        2-tuple of:
        - ``enriched``: List of enriched dicts ready for Phase 3.
        - ``report``:   ``AuditReport`` with full quality metrics.
    """
    # Flatten: accept either a list of raw media dicts or a list of page responses
    flat_media: list[dict[str, Any]] = []
    for item in raw_pages:
        if isinstance(item, list):
            flat_media.extend(item)
        elif isinstance(item, dict) and "data" in item:
            # Raw page snapshot format from storage.py
            page_block = item.get("data", {}).get("Page", {})
            flat_media.extend(page_block.get("media", []))
        else:
            flat_media.append(item)

    logger.info(f"Phase 2 pipeline: processing {len(flat_media)} raw media records.")

    # Step 12 — Sanitize
    catalog, errors = sanitize_raw_media_batch(flat_media)
    if errors:
        logger.warning(
            f"Sanitization skipped {len(errors)} malformed records "
            f"(indices: {[i for i, _ in errors[:10]]}{'...' if len(errors) > 10 else ''})"
        )

    # Step 13 — Franchise graph
    root_map = build_root_id_map(catalog)
    logger.info(
        f"Franchise graph resolved {len(set(root_map.values()))} unique franchise roots "
        f"across {len(catalog)} shows."
    )

    # Step 14 — Audit
    auditor = DataQualityAuditor(catalog, root_map=root_map)
    report = auditor.run_audit(raise_on_critical=raise_on_critical)

    # Enrich with franchise_root_id
    enriched = inject_franchise_roots(catalog, root_map)

    return enriched, report
