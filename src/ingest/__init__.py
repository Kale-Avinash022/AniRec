"""AniRec data ingestion module.

Includes AniList GraphQL client, polite rate-limiting, and raw disk snapshot storage.
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

__all__ = [
    "ANILIST_GRAPHQL_URL",
    "ANILIST_MEDIA_PAGE_QUERY",
    "AniListClient",
    "AsyncRateLimiter",
    "build_page_variables",
    "get_snapshot_path",
    "list_existing_snapshots",
    "load_page_snapshot",
    "save_page_snapshot",
]
