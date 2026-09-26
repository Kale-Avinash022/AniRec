"""Tests for API Latency Profiling & Telemetry (Step 43)."""

import pytest
import time
from unittest.mock import patch
from src.api.telemetry import PipelineProfiler


def test_pipeline_profiler_stages():
    profiler = PipelineProfiler(request_id="req-test-123", endpoint="/recommend")

    profiler.start_stage("parsing")
    time.sleep(0.01)  # 10ms
    duration_parsing = profiler.end_stage("parsing")
    assert duration_parsing >= 8.0

    profiler.start_stage("retrieval")
    time.sleep(0.01)
    duration_retrieval = profiler.end_stage("retrieval")
    assert duration_retrieval >= 8.0

    profiler.record_metric("candidates_found", 20)

    with patch("src.core.logger.logger.log_request_event") as mock_log:
        event = profiler.finish_and_log()
        assert event["request_id"] == "req-test-123"
        assert "parsing" in event["stage_latencies_ms"]
        assert "retrieval" in event["stage_latencies_ms"]
        assert "total" in event["stage_latencies_ms"]
        assert event["metadata"]["candidates_found"] == 20
        mock_log.assert_called_once()
