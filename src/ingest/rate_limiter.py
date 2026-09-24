"""API Rate Limiter and Throttling Engine (Step 09).

Implements an asynchronous token-bucket rate limiter and dynamic header-based
backoff mechanism to respect AniList API rate limits (90 req/min).
"""

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from src.core.logger import logger


class AsyncRateLimiter:
    """Asynchronous token bucket rate limiter for AniList API crawling.

    Default settings allow 80 requests per 60.0 seconds to safely stay below
    AniList's 90 requests/minute enforcement limit.
    """

    def __init__(self, max_rate: int = 80, time_period: float = 60.0) -> None:
        """Initialize the rate limiter.

        Args:
            max_rate: Maximum number of permits allowed per time_period.
            time_period: Window duration in seconds.
        """
        if max_rate <= 0:
            raise ValueError(f"max_rate must be positive, got {max_rate}")
        if time_period <= 0:
            raise ValueError(f"time_period must be positive, got {time_period}")

        self.max_rate = max_rate
        self.time_period = time_period
        self.tokens = float(max_rate)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a request permit, sleeping asynchronously if token bucket is depleted."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_refill

                # Refill tokens based on elapsed time
                refill_amount = elapsed * (self.max_rate / self.time_period)
                self.tokens = min(float(self.max_rate), self.tokens + refill_amount)
                self.last_refill = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

                # Calculate sleep duration needed to replenish at least 1 token
                sleep_duration = (1.0 - self.tokens) * (
                    self.time_period / self.max_rate
                )
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)

    async def handle_rate_limit_headers(self, headers: Mapping[str, Any]) -> None:
        """Inspect response headers from AniList and trigger preemptive sleep if needed.

        Args:
            headers: HTTP response headers dictionary/mapping.
        """
        # AniList sends headers case-insensitively, e.g., 'X-RateLimit-Remaining', 'Retry-After'
        headers_lower = {k.lower(): v for k, v in headers.items()}

        raw_remaining = headers_lower.get("x-ratelimit-remaining")
        if raw_remaining is not None:
            try:
                remaining = int(raw_remaining)
                if remaining < 5:
                    raw_retry_after = headers_lower.get("retry-after", "10")
                    retry_after = max(int(raw_retry_after), 5)
                    logger.warning(
                        f"AniList rate limit nearly exhausted ({remaining} remaining). "
                        f"Backing off for {retry_after}s..."
                    )
                    await asyncio.sleep(retry_after)
            except (ValueError, TypeError):
                pass
