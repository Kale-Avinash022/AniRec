"""
Unit Tests for Step 01: Project Philosophy and SLO Metrics Specification.
"""

import unittest

from src.core.metrics_spec import SKILL_MAPPING, TargetSLO, validate_evaluation_metrics


class TestStep01PhilosophyAndMetrics(unittest.TestCase):
    def setUp(self):
        self.slo = TargetSLO()

    def test_slo_defaults(self):
        """Verify baseline targets match Step 01 specification."""
        self.assertEqual(self.slo.min_hit_rate_at_5, 0.70)
        self.assertEqual(self.slo.min_faithfulness_score, 0.85)
        self.assertEqual(self.slo.max_p95_latency_seconds, 6.0)
        self.assertEqual(self.slo.max_ttft_seconds, 1.5)
        self.assertEqual(self.slo.max_hallucination_rate, 0.0)
        self.assertLessEqual(self.slo.max_cost_per_query_usd, 0.005)

    def test_skill_mapping_completeness(self):
        """Ensure all 8 core engineering competencies are documented."""
        self.assertEqual(len(SKILL_MAPPING), 8)
        components = [s.component for s in SKILL_MAPPING]
        self.assertIn("AniList Ingestion Pipeline", components)
        self.assertIn("Document Design & Chunking", components)
        self.assertIn("Hybrid Retrieval", components)
        self.assertIn("Reranking & Diversity (MMR)", components)
        self.assertIn("Grounded Generation & Guardrails", components)
        self.assertIn("Evals Harness & LLM-as-Judge", components)
        self.assertIn("FastAPI + SSE Streaming", components)
        self.assertIn("Docker & Cloud Deployment", components)

    def test_validate_evaluation_metrics_success(self):
        """Test validation when all metrics satisfy SLO targets."""
        passing_metrics = {
            "hit_rate_at_5": 0.73,
            "faithfulness_score": 0.88,
            "p95_latency_seconds": 4.7,
            "hallucination_rate": 0.0,
            "cost_per_query_usd": 0.00035,
        }
        passed, report = validate_evaluation_metrics(passing_metrics, self.slo)
        self.assertTrue(passed)
        for key, status in report.items():
            self.assertTrue(status.startswith("PASS:"), f"{key} should pass")

    def test_validate_evaluation_metrics_failure(self):
        """Test validation when metrics fail to meet SLO targets."""
        failing_metrics = {
            "hit_rate_at_5": 0.58,  # Below 0.70
            "faithfulness_score": 0.80,  # Below 0.85
            "p95_latency_seconds": 7.5,  # Above 6.0s
            "hallucination_rate": 0.05,  # Above 0.0%
            "cost_per_query_usd": 0.010,  # Above $0.005
        }
        passed, report = validate_evaluation_metrics(failing_metrics, self.slo)
        self.assertFalse(passed)
        self.assertTrue(report["hit_rate_at_5"].startswith("FAIL:"))
        self.assertTrue(report["faithfulness_score"].startswith("FAIL:"))
        self.assertTrue(report["p95_latency_seconds"].startswith("FAIL:"))
        self.assertTrue(report["hallucination_rate"].startswith("FAIL:"))
        self.assertTrue(report["cost_per_query_usd"].startswith("FAIL:"))


if __name__ == "__main__":
    unittest.main()
