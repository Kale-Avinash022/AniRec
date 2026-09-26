"""Unit tests for Step 53: Eval Benchmark Comparison Reporter."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.eval.reporter import (
    generate_comparison_summary,
    generate_comparison_table,
    load_all_eval_reports,
    save_comparison_markdown,
)


def test_load_all_eval_reports_empty_and_nonexistent(tmp_path: Path):
    """Test loading reports from nonexistent or empty folder."""
    assert load_all_eval_reports(tmp_path / "nonexistent") == []
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert load_all_eval_reports(empty_dir) == []


def test_generate_comparison_table_and_summary(tmp_path: Path):
    """Test generating comparison table and summary from multiple run JSON files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    run1 = {
        "summary": {
            "experiment_name": "01_baseline_vector_only",
            "total_queries": 50,
            "hit_rate_at_5": 0.52,
            "mrr": 0.31,
            "avg_faithfulness_score": 4.1,
            "faithfulness_rate": 0.82,
            "constraint_satisfaction_rate": 0.70,
            "p50_latency_ms": 120.0,
            "p95_latency_ms": 280.0,
        },
        "details": [],
    }
    run2 = {
        "summary": {
            "experiment_name": "02_with_reranker_and_dedup",
            "total_queries": 50,
            "hit_rate_at_5": 0.76,
            "mrr": 0.58,
            "avg_faithfulness_score": 4.8,
            "faithfulness_rate": 0.94,
            "constraint_satisfaction_rate": 0.92,
            "p50_latency_ms": 190.0,
            "p95_latency_ms": 410.0,
        },
        "details": [],
    }

    with open(results_dir / "eval_run1.json", "w", encoding="utf-8") as f:
        json.dump(run1, f)
    with open(results_dir / "eval_run2.json", "w", encoding="utf-8") as f:
        json.dump(run2, f)

    # 1. Test loading
    reports = load_all_eval_reports(results_dir)
    assert len(reports) == 2

    # 2. Test comparison table
    table_md = generate_comparison_table(results_dir=results_dir)
    assert "01_baseline_vector_only" in table_md
    assert "02_with_reranker_and_dedup" in table_md
    assert "52.0%" in table_md
    assert "76.0%" in table_md
    assert "94.0%" in table_md

    # 3. Test summary stats
    summary = generate_comparison_summary(results_dir=results_dir)
    assert summary["total_runs"] == 2
    assert summary["best_hit_rate_at_5"] == 0.76
    assert summary["best_mrr"] == 0.58
    assert summary["best_faithfulness_rate"] == 0.94

    # 4. Test saving markdown file
    out_md = tmp_path / "COMPARISON.md"
    saved_path = save_comparison_markdown(output_path=out_md, results_dir=results_dir)
    assert saved_path.exists()
    with open(saved_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "# AniRec Evaluation Benchmark Comparison Report" in content
        assert "02_with_reranker_and_dedup" in content


def test_cli_comparison_runner(tmp_path: Path):
    """Test CLI scripts/compare_evals.py command execution."""
    result = subprocess.run(
        [sys.executable, "scripts/compare_evals.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Render AniRec Evaluation Comparison Table" in result.stdout
