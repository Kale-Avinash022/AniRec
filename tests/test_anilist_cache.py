"""Tests for AniList Persistent Two-Tier Cache & Outage Resilience."""

import time
from pathlib import Path
import pytest

from src.ingest.cache import AniListCache


def test_cache_set_and_get(tmp_path: Path):
    cache = AniListCache(cache_dir=tmp_path, default_ttl=172800)
    data = {"id": 16498, "title": "Attack on Titan", "score": 85}

    cache.set("test_show", data)

    # 1. Retrieve from memory
    hit = cache.get("test_show")
    assert hit is not None
    assert hit["id"] == 16498
    assert hit["title"] == "Attack on Titan"


def test_cache_disk_persistence(tmp_path: Path):
    cache1 = AniListCache(cache_dir=tmp_path, default_ttl=172800)
    data = {"id": 101922, "title": "Demon Slayer"}
    cache1.set("anime_101922", data)

    # 2. Simulate process restart with fresh cache instance (empty memory)
    cache2 = AniListCache(cache_dir=tmp_path, default_ttl=172800)
    hit = cache2.get("anime_101922")
    assert hit is not None
    assert hit["id"] == 101922
    assert hit["title"] == "Demon Slayer"


def test_cache_ttl_and_stale_fallback(tmp_path: Path):
    # Short TTL for test: 1 second
    cache = AniListCache(cache_dir=tmp_path, default_ttl=1)
    data = {"id": 20, "title": "Naruto"}
    cache.set("anime_20", data, ttl_seconds=1)

    # Immediate check
    assert cache.get("anime_20", allow_stale_on_outage=False) is not None

    # Wait for TTL expiry
    time.sleep(1.1)

    # Strict check (no stale allowed) should return None
    assert cache.get("anime_20", allow_stale_on_outage=False) is None

    # Outage fallback check (allow stale) should return cached data gracefully
    stale_fallback = cache.get("anime_20", allow_stale_on_outage=True)
    assert stale_fallback is not None
    assert stale_fallback["title"] == "Naruto"


def test_cache_stats(tmp_path: Path):
    cache = AniListCache(cache_dir=tmp_path, default_ttl=172800)
    cache.set("item_1", {"foo": "bar"})
    cache.set("item_2", {"baz": "qux"})

    stats = cache.stats()
    assert stats["memory_items"] == 2
    assert stats["disk_files"] == 2
    assert stats["default_ttl_hours"] == 48
