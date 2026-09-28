"""Fault-Tolerant AniList GraphQL Ingestion Client (Step 11).

Provides paginated harvesting, 2-day persistent caching, rate limiting, and exponential
backoff retry handling to crawl public anime data from AniList.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from src.core.config import settings
from src.core.logger import logger
from src.ingest.cache import AniListCache, get_anilist_cache
from src.ingest.queries import (
    ANILIST_GRAPHQL_URL,
    ANILIST_MEDIA_PAGE_QUERY,
    build_page_variables,
)
from src.ingest.rate_limiter import AsyncRateLimiter
from src.ingest.storage import load_page_snapshot, save_page_snapshot


class AniListClient:
    """Async GraphQL client for scraping, caching, and serving anime records from AniList."""

    def __init__(
        self,
        api_url: str | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        data_dir: Path | None = None,
        cache: AniListCache | None = None,
    ) -> None:
        """Initialize the client."""
        self.url = (
            api_url
            if api_url is not None
            else settings.ANILIST_API_URL or ANILIST_GRAPHQL_URL
        )
        self.rate_limiter = (
            rate_limiter if rate_limiter is not None else AsyncRateLimiter()
        )
        self.data_dir = data_dir if data_dir is not None else settings.RAW_DATA_DIR
        self.cache = (
            cache
            if cache is not None
            else (
                AniListCache(cache_dir=self.data_dir / "cache")
                if data_dir is not None
                else get_anilist_cache()
            )
        )

    async def fetch_page(
        self,
        page: int,
        per_page: int = 50,
        max_retries: int = 5,
        use_cache: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Fetch a single page of anime records with 2-day TTL cache and outage fallback."""
        cache_key = f"page_{page:03d}"

        if use_cache:
            # Check raw disk snapshot in specified data_dir first
            raw_snapshot = load_page_snapshot(page, data_dir=self.data_dir)
            if raw_snapshot is not None:
                logger.debug(f"Loaded page {page} from disk snapshot.")
                return raw_snapshot

            # Check 2-day persistent cache
            cached_page = self.cache.get(cache_key, allow_stale_on_outage=False)
            if cached_page is not None:
                logger.debug(f"Loaded page {page} from 2-day persistent cache.")
                return cached_page

        variables = build_page_variables(page=page, per_page=per_page)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "query": ANILIST_MEDIA_PAGE_QUERY,
            "variables": variables,
        }

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
                        self.cache.set(cache_key, data)
                    logger.info(
                        f"Successfully fetched page {page} from AniList (attempt {attempt})."
                    )
                    return data

                if resp.status_code == 429:
                    retry_header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
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
                    if use_cache:
                        fallback_cached = self.cache.get(cache_key, allow_stale_on_outage=True)
                        if fallback_cached is not None:
                            logger.warning(f"AniList unreachable; serving fallback cached data for page {page}.")
                            return fallback_cached

                        raw_fallback = load_page_snapshot(page, data_dir=self.data_dir)
                        if raw_fallback is not None:
                            logger.warning(f"AniList unreachable; serving raw disk snapshot for page {page}.")
                            return raw_fallback

                    raise RuntimeError(
                        f"Failed to fetch page {page} after {max_retries} attempts."
                    ) from exc
                await asyncio.sleep(delay)
            finally:
                if owns_client:
                    await http_client.aclose()

        if use_cache:
            fallback_data = self.cache.get(cache_key, allow_stale_on_outage=True) or load_page_snapshot(page, data_dir=self.data_dir)
            if fallback_data is not None:
                logger.warning(f"AniList unavailable; returned cached fallback for page {page}.")
                return fallback_data

        raise RuntimeError(f"Failed to fetch page {page} after {max_retries} attempts.")

    async def ingest_catalog(
        self,
        max_pages: int = 60,
        per_page: int = 50,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Paginate through AniList catalog and collect all anime media entries."""
        all_shows: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=25.0) as client:
            for page in range(1, max_pages + 1):
                try:
                    page_data = await self.fetch_page(
                        page=page,
                        per_page=per_page,
                        use_cache=use_cache,
                        client=client,
                    )
                except Exception as exc:
                    logger.warning(f"Skipping unreachable page {page} during catalog ingestion: {exc}")
                    continue

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
