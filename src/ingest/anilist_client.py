"""Fault-Tolerant AniList GraphQL Ingestion Client (Step 11).

Provides paginated harvesting, disk caching, rate limiting, and exponential
backoff retry handling to crawl public anime data from AniList.
"""

import asyncio
from pathlib import Path
from typing import Any

import httpx

from src.core.config import settings
from src.core.logger import logger
from src.ingest.queries import (
    ANILIST_GRAPHQL_URL,
    ANILIST_MEDIA_PAGE_QUERY,
    build_page_variables,
)
from src.ingest.rate_limiter import AsyncRateLimiter
from src.ingest.storage import load_page_snapshot, save_page_snapshot


class AniListClient:
    """Async GraphQL client for scraping and caching anime records from AniList."""

    def __init__(
        self,
        api_url: str | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        data_dir: Path | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_url: GraphQL endpoint URL (defaults to configured settings).
            rate_limiter: Rate limiter instance; defaults to new AsyncRateLimiter.
            data_dir: Directory where raw snapshot files are saved.
        """
        self.url = (
            api_url
            if api_url is not None
            else settings.ANILIST_API_URL or ANILIST_GRAPHQL_URL
        )
        self.rate_limiter = (
            rate_limiter if rate_limiter is not None else AsyncRateLimiter()
        )
        self.data_dir = data_dir if data_dir is not None else settings.RAW_DATA_DIR

    async def fetch_page(
        self,
        page: int,
        per_page: int = 50,
        max_retries: int = 5,
        use_cache: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Fetch a single page of anime records, checking disk cache first.

        Args:
            page: 1-indexed page number.
            per_page: Number of items per page (maximum 50).
            max_retries: Number of retry attempts on 429, 5xx, or network failure.
            use_cache: Whether to return and write to disk snapshots.
            client: Optional externally-managed httpx.AsyncClient instance.

        Returns:
            Dictionary containing the raw AniList GraphQL response.

        Raises:
            RuntimeError: If all retry attempts are exhausted without success.
        """
        if use_cache:
            cached = load_page_snapshot(page, data_dir=self.data_dir)
            if cached is not None:
                logger.debug(f"Loaded page {page} from disk snapshot cache.")
                return cached

        variables = build_page_variables(page=page, per_page=per_page)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "query": ANILIST_MEDIA_PAGE_QUERY,
            "variables": variables,
        }

        # Use passed client or manage a local client
        owns_client = client is None

        for attempt in range(1, max_retries + 1):
            await self.rate_limiter.acquire()
            http_client = (
                client if client is not None else httpx.AsyncClient(timeout=25.0)
            )

            try:
                resp = await http_client.post(self.url, json=payload, headers=headers)
                await self.rate_limiter.handle_rate_limit_headers(resp.headers)

                if resp.status_code == 200:
                    data: dict[str, Any] = resp.json()
                    if use_cache:
                        save_page_snapshot(page, data, data_dir=self.data_dir)
                    logger.info(
                        f"Successfully fetched page {page} (attempt {attempt})."
                    )
                    return data

                if resp.status_code == 429:
                    retry_header = resp.headers.get("retry-after") or resp.headers.get(
                        "Retry-After"
                    )
                    delay = (
                        int(retry_header)
                        if retry_header and retry_header.isdigit()
                        else (2**attempt)
                    )
                    logger.warning(
                        f"Rate limited (429) on page {page}. Sleeping {delay}s (attempt {attempt}/{max_retries})."
                    )
                    await asyncio.sleep(delay)
                elif resp.status_code >= 500:
                    delay = 2**attempt
                    logger.warning(
                        f"Server error {resp.status_code} on page {page}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Unexpected HTTP {resp.status_code} on page {page}: {resp.text}"
                    )
                    resp.raise_for_status()

            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                delay = 2**attempt
                logger.error(
                    f"Network error on page {page} (attempt {attempt}/{max_retries}): {exc}"
                )
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to fetch page {page} after {max_retries} attempts."
                    ) from exc
                await asyncio.sleep(delay)
            finally:
                if owns_client:
                    await http_client.aclose()

        raise RuntimeError(f"Failed to fetch page {page} after {max_retries} attempts.")

    async def ingest_catalog(
        self,
        max_pages: int = 60,
        per_page: int = 50,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Paginate through AniList catalog and collect all anime media entries.

        Args:
            max_pages: Maximum pages to crawl (e.g. 60 pages * 50 = 3,000 anime).
            per_page: Number of shows per page.
            use_cache: Whether to use disk snapshot cache.

        Returns:
            List of all extracted media dictionaries.
        """
        all_shows: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=25.0) as client:
            for page in range(1, max_pages + 1):
                page_data = await self.fetch_page(
                    page=page,
                    per_page=per_page,
                    use_cache=use_cache,
                    client=client,
                )

                page_block = page_data.get("data", {}).get("Page", {})
                media_list = page_block.get("media", [])
                all_shows.extend(media_list)

                page_info = page_block.get("pageInfo", {})
                has_next = page_info.get("hasNextPage", False)

                logger.info(
                    f"Ingested page {page}/{max_pages}: +{len(media_list)} shows (total: {len(all_shows)})"
                )

                if not has_next:
                    logger.info(f"Reached end of AniList catalog at page {page}.")
                    break

        logger.info(
            f"Ingestion complete: Collected {len(all_shows)} total anime records."
        )
        return all_shows
