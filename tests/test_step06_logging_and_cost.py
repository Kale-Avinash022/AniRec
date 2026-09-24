"""
Unit Tests for Step 06: Logging, Structured Tracing & Observability Foundations.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.core.cost import calculate_query_cost
from src.core.logger import StageTimer, StructuredLogger


class TestStep06LoggingAndCost(unittest.TestCase):
    def test_cost_calculation_gpt4o_mini(self):
        """Verify cost calculation for OpenAI gpt-4o-mini."""
        # 1,000 prompt tokens ($0.15/1M) = $0.00015
        # 500 completion tokens ($0.60/1M) = $0.00030
        # Total = $0.00045
        cost = calculate_query_cost(
            "gpt-4o-mini", prompt_tokens=1000, completion_tokens=500
        )
        self.assertEqual(cost, 0.00045)

    def test_cost_calculation_embeddings(self):
        """Verify cost calculation for text-embedding-3-small."""
        # 5,000 tokens ($0.020/1M) = $0.0001
        cost = calculate_query_cost("text-embedding-3-small", prompt_tokens=5000)
        self.assertEqual(cost, 0.0001)

    def test_cost_calculation_local_models(self):
        """Verify local models calculate $0.00 cost."""
        cost_ollama = calculate_query_cost(
            "llama3.1:8b", prompt_tokens=2000, completion_tokens=500
        )
        self.assertEqual(cost_ollama, 0.0)

        cost_st = calculate_query_cost(
            "sentence-transformers/all-MiniLM-L6-v2", prompt_tokens=2000
        )
        self.assertEqual(cost_st, 0.0)

    def test_stage_timer_latency_tracking(self):
        """Verify StageTimer correctly records stage intervals and total latency."""
        timer = StageTimer()
        with timer.measure("stage_one"):
            time.sleep(0.01)  # 10ms

        with timer.measure("stage_two"):
            time.sleep(0.01)  # 10ms

        latencies = timer.finish()
        self.assertIn("stage_one", latencies)
        self.assertIn("stage_two", latencies)
        self.assertIn("total", latencies)
        self.assertGreaterEqual(latencies["stage_one"], 5)
        self.assertGreaterEqual(latencies["stage_two"], 5)
        self.assertGreaterEqual(
            latencies["total"], latencies["stage_one"] + latencies["stage_two"]
        )

    def test_structured_logger_jsonl_output(self):
        """Verify StructuredLogger writes valid JSONL records with timestamps and schema."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            logger_instance = StructuredLogger(
                log_filename="test_traces.jsonl", log_dir=log_dir
            )

            event = {
                "request_id": "req_test_123",
                "user_query": "Steins;Gate thriller",
                "parsed_filters": {"genres": ["Sci-Fi", "Thriller"]},
                "stage_latencies_ms": {"total": 450, "vector_retrieval": 50},
                "tokens": {
                    "prompt_tokens": 800,
                    "completion_tokens": 150,
                    "total_tokens": 950,
                },
                "estimated_cost_usd": 0.00021,
                "hallucination_detected": False,
            }

            logger_instance.log_request_event(event)

            log_file = log_dir / "test_traces.jsonl"
            self.assertTrue(log_file.exists())

            lines = log_file.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)

            logged_record = json.loads(lines[0])
            self.assertEqual(logged_record["request_id"], "req_test_123")
            self.assertIn("timestamp", logged_record)
            self.assertEqual(logged_record["stage_latencies_ms"]["total"], 450)
            self.assertEqual(logged_record["estimated_cost_usd"], 0.00021)
            self.assertFalse(logged_record["hallucination_detected"])


if __name__ == "__main__":
    unittest.main()
