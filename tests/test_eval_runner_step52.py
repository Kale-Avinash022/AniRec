"""Unit tests for Step 52: Automated Evals Harness Runner."""

import json
from pathlib import Path
import pytest

from src.eval.runner import run_evaluation_suite
from src.generation.llm_client import MockChatModel


@pytest.mark.asyncio
async def test_run_evaluation_suite_sample_run(tmp_path: Path):
    """Test running evaluation suite on a small sample subset with Mock LLM."""
    mock_llm = MockChatModel(
        response_text=json.dumps({
            "recommendations": [
                {
                    "anilist_id": 101348,
                    "title": "Vinland Saga",
                    "year": 2019,
                    "why": "Dark historical warfare with profound political intrigue and morally grey themes.",
                    "matched_attributes": ["Politics", "Military", "Historical"],
                }
            ],
            "reasoning_summary": "Top grounded anime matches for your thematic request.",
        })
    )


    out_dir = tmp_path / "eval_results"
    report = await run_evaluation_suite(
        experiment_name="unit_test_run",
        max_samples=2,
        out_dir=out_dir,
        llm=mock_llm,
        verbose=False,
    )

    # Validate report structure
    assert "summary" in report
    assert "details" in report

    summary = report["summary"]
    assert summary["experiment_name"] == "unit_test_run"
    assert summary["total_queries"] == 2
    assert "hit_rate_at_5" in summary
    assert "mrr" in summary
    assert "faithfulness_rate" in summary
    assert "p50_latency_ms" in summary
    assert "p95_latency_ms" in summary

    details = report["details"]
    assert len(details) == 2
    for record in details:
        assert "id" in record
        assert "query" in record
        assert "hit_at_5" in record
        assert "mrr" in record
        assert "faithfulness_score" in record
        assert "latency_ms" in record

    # Validate disk file was written
    written_files = list(out_dir.glob("eval_unit_test_run_*.json"))
    assert len(written_files) == 1

    with open(written_files[0], "r", encoding="utf-8") as f:
        disk_data = json.load(f)
        assert disk_data["summary"]["experiment_name"] == "unit_test_run"
        assert len(disk_data["details"]) == 2


def test_cli_runner_help():
    """Verify scripts/run_evals.py entrypoint is importable and executes help without errors."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/run_evals.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run AniRec Evaluation Benchmark Suite" in result.stdout

