"""Unit tests for Step 54: Iterative Optimization Experiments & Baseline Progression."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.eval.experiments import run_ablation_experiment, run_all_ablations
from src.generation.llm_client import MockChatModel


@pytest.fixture
def mock_chat_llm():
    return MockChatModel(
        response_text=json.dumps({
            "recommendations": [
                {
                    "anilist_id": 101348,
                    "title": "Vinland Saga",
                    "year": 2019,
                    "why": "Grim medieval political warfare and complex character development.",
                    "matched_attributes": ["Action", "Adventure", "Drama"],
                }
            ],
            "reasoning_summary": "Top recommendations matching your query.",
        })
    )


@pytest.mark.asyncio
async def test_ablation_baseline_run(tmp_path: Path, mock_chat_llm):
    """Test running 01_baseline_vector_only ablation mode."""
    out_dir = tmp_path / "evals"
    report = await run_ablation_experiment(
        mode="01_baseline_vector_only",
        max_samples=2,
        out_dir=out_dir,
        llm=mock_chat_llm,
        verbose=False,
    )
    assert report["summary"]["experiment_name"] == "01_baseline_vector_only"
    assert report["summary"]["total_queries"] == 2
    assert "hit_rate_at_5" in report["summary"]
    assert "mrr" in report["summary"]
    assert (out_dir / "eval_01_baseline_vector_only.json").exists()


@pytest.mark.asyncio
async def test_ablation_cross_encoder_run(tmp_path: Path, mock_chat_llm):
    """Test running 02_with_cross_encoder ablation mode."""
    out_dir = tmp_path / "evals"
    report = await run_ablation_experiment(
        mode="02_with_cross_encoder",
        max_samples=2,
        out_dir=out_dir,
        llm=mock_chat_llm,
        verbose=False,
    )
    assert report["summary"]["experiment_name"] == "02_with_cross_encoder"
    assert report["summary"]["total_queries"] == 2
    assert (out_dir / "eval_02_with_cross_encoder.json").exists()


@pytest.mark.asyncio
async def test_ablation_franchise_dedup_run(tmp_path: Path, mock_chat_llm):
    """Test running 03_with_franchise_dedup ablation mode."""
    out_dir = tmp_path / "evals"
    report = await run_ablation_experiment(
        mode="03_with_franchise_dedup",
        max_samples=2,
        out_dir=out_dir,
        llm=mock_chat_llm,
        verbose=False,
    )
    assert report["summary"]["experiment_name"] == "03_with_franchise_dedup"
    assert report["summary"]["total_queries"] == 2
    assert (out_dir / "eval_03_with_franchise_dedup.json").exists()


@pytest.mark.asyncio
async def test_ablation_full_production_run(tmp_path: Path, mock_chat_llm):
    """Test running 04_full_production_guardrails ablation mode."""
    out_dir = tmp_path / "evals"
    report = await run_ablation_experiment(
        mode="04_full_production_guardrails",
        max_samples=2,
        out_dir=out_dir,
        llm=mock_chat_llm,
        verbose=False,
    )
    assert report["summary"]["experiment_name"] == "04_full_production_guardrails"
    assert report["summary"]["total_queries"] == 2
    assert (out_dir / "eval_04_full_production_guardrails.json").exists()


def test_cli_ablation_runner():
    """Verify scripts/run_ablation_benchmarks.py CLI entrypoint executes help without errors."""
    result = subprocess.run(
        [sys.executable, "scripts/run_ablation_benchmarks.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run Full Ablation Benchmark Progression" in result.stdout
