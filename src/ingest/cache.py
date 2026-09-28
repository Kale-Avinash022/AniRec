"""AniList Two-Tier Persistent Cache & Offline Resilience Layer.

Provides:
  1. In-memory fast tier (L1) for sub-millisecond hot lookups.
  2. Persistent atomic disk cache (L2) with configurable TTL (default 2 days / 48 hours).
  3. Graceful offline fallback: if AniList is unreachable, down, or throttled,
     serves existing cached snapshots and catalog data without failing user requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logger import logger


class AniListCache:
    """Persistent two-tier cache with TTL expiration and stale-fallback capability."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        default_ttl: int | None = None,
    ) -> None:
        self.cache_dir = cache_dir if cache_dir is not None else settings.CACHE_DIR
        self.default_ttl = default_ttl if default_ttl is not None else settings.ANILIST_CACHE_TTL_SECONDS
        self._memory_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, key_or_query: str) -> str:
        """Create safe filename hash from key or query string."""
        clean = key_or_query.strip()
        if clean.isalnum() and len(clean) <= 40:
            return clean
        h = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:24]
        return f"cache_{h}"

    def get(
        self,
        key: str,
        allow_stale_on_outage: bool = True,
    ) -> dict[str, Any] | None:
        """Retrieve cached payload from L1 memory or L2 disk cache.

        Args:
            key: Unique lookup key (e.g. 'anime_16498' or 'page_001').
            allow_stale_on_outage: If True, returns cached record even if TTL expired.

        Returns:
            Cached payload dict or None if not found.
        """
        now = time.time()
        file_key = self._make_key(key)

        # ── 1. Check L1 In-Memory Cache ───────────────────────────────────────
        if file_key in self._memory_cache:
            expiry, data = self._memory_cache[file_key]
            if now <= expiry:
                return data
            if allow_stale_on_outage:
                logger.debug(f"Serving expired in-memory cache for key '{key}' (outage fallback)")
                return data

        # ── 2. Check L2 Persistent Disk Cache ────────────────────────────────
        disk_path = self.cache_dir / f"{file_key}.json"
        if not disk_path.exists():
            return None

        try:
            with open(disk_path, encoding="utf-8") as f:
                envelope: dict[str, Any] = json.load(f)

            expiry = envelope.get("expires_at", 0.0)
            data = envelope.get("data")
            if data is None:
                return None

            # Promote to L1 memory cache
            self._memory_cache[file_key] = (expiry, data)

            if now <= expiry:
                logger.debug(f"Cache hit on disk for key '{key}' (TTL remaining: {int(expiry - now)}s)")
                return data

            if allow_stale_on_outage:
                logger.info(f"Serving stale disk cache for '{key}' (expired {int(now - expiry)}s ago)")
                return data

            return None
        except Exception as exc:
            logger.warning(f"Error reading disk cache for key '{key}': {exc}")
            return None

    def set(
        self,
        key: str,
        data: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Store payload into both L1 memory cache and L2 atomic disk cache.

        Args:
            key: Unique key identifier.
            data: Data payload to cache.
            ttl_seconds: Cache validity duration in seconds (defaults to 2 days).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()
        expiry = now + ttl
        file_key = self._make_key(key)

        # 1. Store in L1 memory
        self._memory_cache[file_key] = (expiry, data)

        # 2. Persist atomically to L2 disk
        envelope = {
            "key": key,
            "created_at": now,
            "expires_at": expiry,
            "ttl_seconds": ttl,
            "data": data,
        }

        target_path = self.cache_dir / f"{file_key}.json"
        temp_path = target_path.with_suffix(".tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, target_path)
            logger.debug(f"Cached '{key}' to disk (valid for {ttl // 3600}h)")
        except Exception as exc:
            logger.error(f"Failed to persist cache for key '{key}': {exc}")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def get_anime(self, anime_id: int) -> dict[str, Any] | None:
        """Get cached anime detail dictionary by ID."""
        return self.get(f"anime_{anime_id}")

    def set_anime(
        self,
        anime_id: int,
        data: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache single anime detail record."""
        self.set(f"anime_{anime_id}", data, ttl_seconds=ttl_seconds)

    def get_page(self, page: int) -> dict[str, Any] | None:
        """Get cached catalog page by page number."""
        return self.get(f"page_{page:03d}")

    def set_page(
        self,
        page: int,
        data: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Cache catalog page by page number."""
        self.set(f"page_{page:03d}", data, ttl_seconds=ttl_seconds)

    def stats(self) -> dict[str, Any]:
        """Return cache health metrics."""
        disk_files = list(self.cache_dir.glob("*.json"))
        return {
            "memory_items": len(self._memory_cache),
            "disk_files": len(disk_files),
            "cache_dir": str(self.cache_dir),
            "default_ttl_hours": self.default_ttl // 3600,
        }


# Global singleton instance
_anilist_cache: AniListCache | None = None


def get_anilist_cache() -> AniListCache:
    """Retrieve or initialize the global AniList cache singleton."""
    global _anilist_cache
    if _anilist_cache is None:
        _anilist_cache = AniListCache()
    return _anilist_cache
