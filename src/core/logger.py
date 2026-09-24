"""
Structured Logging, Request Tracing and Observability Module (Step 06).
Records query latency, token usage, monetary cost, and audit traces in JSONL format.
"""

import json
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.config import settings


class StageTimer:
    """Helper class to track individual stage latencies during request processing."""

    def __init__(self) -> None:
        self.stage_latencies_ms: dict[str, int] = {}
        self._start_time: float = time.perf_counter()

    @contextmanager
    def measure(self, stage_name: str) -> Generator[None, None, None]:
        """Context manager to measure execution time of a specific pipeline stage in milliseconds."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self.stage_latencies_ms[stage_name] = elapsed_ms

    def finish(self) -> dict[str, int]:
        """Compute total elapsed latency and return all recorded stage latencies."""
        total_ms = int((time.perf_counter() - self._start_time) * 1000)
        self.stage_latencies_ms["total"] = total_ms
        return self.stage_latencies_ms


class StructuredLogger:
    """Structured JSONL event logger and application console logger."""

    def __init__(
        self, log_filename: str = "app.jsonl", log_dir: Path | None = None
    ) -> None:
        self.log_dir = log_dir if log_dir is not None else settings.LOGS_DIR
        self.log_path = self.log_dir / log_filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Standard console logger
        self.logger = logging.getLogger("anirec")
        self.logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_request_event(self, event_data: dict[str, Any]) -> None:
        """
        Record a comprehensive JSONL trace for a completed request.

        Args:
            event_data: Structured dictionary containing request_id, filters, latencies, tokens, etc.
        """
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now(UTC).isoformat()

        json_line = json.dumps(event_data)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        total_lat = event_data.get("stage_latencies_ms", {}).get("total", 0)
        req_id = event_data.get("request_id", "unknown")
        self.logger.info(f"Processed Request {req_id} in {total_lat}ms")

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


# Global singleton logger instance
logger = StructuredLogger()
