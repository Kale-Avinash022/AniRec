"""API Latency Profiling & Token Cost Tracking Telemetry (Step 43).

Tracks granular execution metrics across all pipeline stages:
  1. Query parsing (Step 25/26)
  2. Vector retrieval & filter relaxation (Steps 27/28)
  3. Cross-Encoder reranking (Step 29)
  4. Grounded generation & verification (Steps 32-37)

Persists structured audit events to ``logs/app.jsonl`` for dashboard analytics.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.logger import logger


class PipelineProfiler:
    """Stage-by-stage latency and telemetry profiler for a single request."""

    def __init__(self, request_id: str, endpoint: str) -> None:
        self.request_id = request_id
        self.endpoint = endpoint
        self.start_time = time.perf_counter()
        self.stage_durations: dict[str, float] = {}
        self._active_stage: str | None = None
        self._stage_start: float = 0.0
        self.metadata: dict[str, Any] = {}

    def start_stage(self, name: str) -> None:
        """Begin measuring a named pipeline stage."""
        self._active_stage = name
        self._stage_start = time.perf_counter()

    def end_stage(self, name: str | None = None) -> float:
        """Finish measuring the named stage and record duration in milliseconds."""
        target_stage = name or self._active_stage
        if target_stage is None or self._stage_start == 0.0:
            return 0.0

        duration_ms = round((time.perf_counter() - self._stage_start) * 1000, 2)
        self.stage_durations[target_stage] = duration_ms
        self._active_stage = None
        self._stage_start = 0.0
        return duration_ms

    def record_metric(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata / metrics (e.g. token cost, candidates count)."""
        self.metadata[key] = value

    def finish_and_log(self) -> dict[str, Any]:
        """Compute total elapsed time and write structured telemetry log."""
        total_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        self.stage_durations["total"] = total_ms

        event_data = {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "stage_latencies_ms": self.stage_durations,
            "metadata": self.metadata,
        }

        try:
            logger.log_request_event(event_data)
        except Exception as exc:
            logger.warning(f"Failed to record request telemetry: {exc}")

        return event_data
