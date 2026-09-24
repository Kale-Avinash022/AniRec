"""Unit tests for Step 09: API Rate Limiter & Throttling Engine."""

import asyncio
import time

import pytest

from src.ingest import AsyncRateLimiter


def test_rate_limiter_initialization() -> None:
    """Verify default and customized initialization parameters."""
    limiter = AsyncRateLimiter(max_rate=80, time_period=60.0)
    assert limiter.max_rate == 80
    assert limiter.time_period == 60.0
    assert limiter.tokens == 80.0

    with pytest.raises(ValueError, match="max_rate must be positive"):
        AsyncRateLimiter(max_rate=0)

    with pytest.raises(ValueError, match="time_period must be positive"):
        AsyncRateLimiter(max_rate=10, time_period=-1.0)


@pytest.mark.asyncio
async def test_rate_limiter_token_consumption() -> None:
    """Verify tokens are correctly consumed upon permit acquisition."""
    limiter = AsyncRateLimiter(max_rate=5, time_period=1.0)

    # Acquire 3 permits immediately
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()

    assert limiter.tokens < 3.0


@pytest.mark.asyncio
async def test_rate_limiter_throttling_delays() -> None:
    """Verify rate limiter pauses when permits are exhausted."""
    # 2 permits per 0.2 seconds -> roughly 0.1s per permit
    limiter = AsyncRateLimiter(max_rate=2, time_period=0.2)

    start_time = time.monotonic()

    # Exhaust available tokens
    await limiter.acquire()
    await limiter.acquire()

    # Third acquire should wait for token replenishment
    await limiter.acquire()

    elapsed = time.monotonic() - start_time
    assert elapsed >= 0.08  # Should have waited for at least one token duration


@pytest.mark.asyncio
async def test_handle_rate_limit_headers_normal() -> None:
    """Verify normal headers with plenty of remaining quota do not block."""
    limiter = AsyncRateLimiter()
    headers = {"X-RateLimit-Remaining": "50", "X-RateLimit-Limit": "90"}

    t0 = time.monotonic()
    await limiter.handle_rate_limit_headers(headers)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_handle_rate_limit_headers_exhausted_backoff() -> None:
    """Verify header with low remaining requests triggers backoff sleep."""
    limiter = AsyncRateLimiter()
    # Mock low remaining with small Retry-After to test quickly
    headers = {
        "x-ratelimit-remaining": "2",
        "retry-after": "0",  # Will use minimum 5s or test logic
    }

    # Verify that headers with remaining < 5 triggers sleep branch
    # We patch asyncio.sleep to verify invocation without waiting 5 real seconds
    sleep_called = False
    original_sleep = asyncio.sleep

    async def mock_sleep(delay: float) -> None:
        nonlocal sleep_called
        sleep_called = True

    try:
        asyncio.sleep = mock_sleep  # type: ignore[assignment]
        await limiter.handle_rate_limit_headers(headers)
        assert sleep_called is True
    finally:
        asyncio.sleep = original_sleep  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_concurrent_permits() -> None:
    """Verify thread/coroutine safety under concurrent task gathering."""
    limiter = AsyncRateLimiter(max_rate=10, time_period=1.0)

    async def worker() -> int:
        await limiter.acquire()
        return 1

    tasks = [worker() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    assert sum(results) == 5
