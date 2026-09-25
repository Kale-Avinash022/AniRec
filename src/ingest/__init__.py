"""AniRec data ingestion module.

Includes AniList GraphQL client, polite rate-limiting, raw disk snapshot
storage, and metadata extraction / field sanitization (Step 12).
"""

from src.ingest.anilist_client import AniListClient
from src.ingest.queries import (
    ANILIST_GRAPHQL_URL,
    ANILIST_MEDIA_PAGE_QUERY,
    build_page_variables,
)
from src.ingest.rate_limiter import AsyncRateLimiter
from src.ingest.storage import (
    get_snapshot_path,
    list_existing_snapshots,
    load_page_snapshot,
    save_page_snapshot,
)
from src.ingest.franchise_graph import (
    FRANCHISE_RELATION_TYPES,
    FranchiseGraph,
    build_franchise_graph,
    build_root_id_map,
)
from src.ingest.transform import (
    RelationEdge,
    SanitizedAnime,
    SanitizedTag,
    clean_html_description,
    sanitize_raw_media,
    sanitize_raw_media_batch,
)

from src.ingest.validate import (
    AuditReport,
    DataQualityAuditor,
    inject_franchise_roots,
    run_full_pipeline,
)

__all__ = [
    # Step 08 — Queries
    "ANILIST_GRAPHQL_URL",
    "ANILIST_MEDIA_PAGE_QUERY",
    "build_page_variables",
    # Step 09 — Rate Limiter
    "AsyncRateLimiter",
    # Step 10 — Storage
    "get_snapshot_path",
    "list_existing_snapshots",
    "load_page_snapshot",
    "save_page_snapshot",
    # Step 11 — Client
    "AniListClient",
    # Step 12 — Transform
    "RelationEdge",
    "SanitizedAnime",
    "SanitizedTag",
    "clean_html_description",
    "sanitize_raw_media",
    "sanitize_raw_media_batch",
    # Step 13 — Franchise Graph
    "FRANCHISE_RELATION_TYPES",
    "FranchiseGraph",
    "build_franchise_graph",
    "build_root_id_map",
    # Step 14 — Validation
    "AuditReport",
    "DataQualityAuditor",
    "inject_franchise_roots",
    "run_full_pipeline",
]
