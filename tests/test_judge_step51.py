"""Unit tests for Step 51: Generation Metrics & LLM-as-a-Judge Faithfulness Harness."""

import pytest
from src.eval.judge import (
    ConstraintSatisfactionVerdict,
    FaithfulnessVerdict,
    evaluate_generation_batch,
    judge_constraint_satisfaction,
    judge_faithfulness,
)
from src.generation.llm_client import MockChatModel


@pytest.mark.asyncio
async def test_judge_faithfulness_passing_verdict():
    """Test LLM-as-judge parsing a high-faithfulness verdict."""
    mock_llm = MockChatModel(
        response_text="""{
            "score": 5,
            "reasoning": "All plot and character points match the retrieved context exactly.",
            "is_faithful": true
        }"""
    )

    context = "Cowboy Bebop is a 1998 space western anime about Spike Spiegel and bounty hunters."
    answer = "I recommend Cowboy Bebop because it follows Spike Spiegel chasing bounties across space in 1998."

    verdict = await judge_faithfulness(context, answer, llm=mock_llm)
    assert verdict.score == 5
    assert verdict.is_faithful is True
    assert "Cowboy Bebop" not in verdict.reasoning or len(verdict.reasoning) > 5


@pytest.mark.asyncio
async def test_judge_faithfulness_failing_verdict():
    """Test LLM-as-judge parsing an unfaithful / hallucinated response."""
    mock_llm = MockChatModel(
        response_text="""{
            "score": 2,
            "reasoning": "Answer claims Spike goes to high school with magical girls, which is completely contradicted by context.",
            "is_faithful": false
        }"""
    )

    context = "Cowboy Bebop is a space western anime."
    answer = "Spike goes to a high school with magical girls and fights demons."

    verdict = await judge_faithfulness(context, answer, llm=mock_llm)
    assert verdict.score == 2
    assert verdict.is_faithful is False


@pytest.mark.asyncio
async def test_judge_faithfulness_empty_inputs():
    """Test edge cases with empty answers or empty contexts."""
    verdict_empty_ans = await judge_faithfulness("Some context", "")
    assert verdict_empty_ans.score == 1
    assert verdict_empty_ans.is_faithful is False

    verdict_empty_ctx = await judge_faithfulness("", "Some answer")
    assert verdict_empty_ctx.score == 1
    assert verdict_empty_ctx.is_faithful is False


@pytest.mark.asyncio
async def test_judge_faithfulness_malformed_json_fallback():
    """Verify robust fallback when LLM returns unstructured text instead of JSON."""
    mock_llm = MockChatModel(
        response_text="I think this is pretty faithful to the context."
    )

    verdict = await judge_faithfulness("Some context", "Valid answer text with good length.", llm=mock_llm)
    assert 1 <= verdict.score <= 5
    assert "Heuristic" in verdict.reasoning or "Parsed" in verdict.reasoning



def test_judge_constraint_satisfaction_pass():
    """Verify constraint checker passes when all candidate attributes comply."""
    candidates = [
        {
            "anime_id": 101,
            "season_year": 2020,
            "format": "TV",
            "average_score": 85,
            "genres": "Action,Sci-Fi,Drama",
        },
        {
            "anime_id": 102,
            "season_year": 2022,
            "format": "TV",
            "average_score": 90,
            "genres": "Action,Sci-Fi",
        },
    ]
    required = {
        "year_min": 2018,
        "format": "TV",
        "min_score": 80,
        "genres": ["Action", "Sci-Fi"],
    }

    verdict = judge_constraint_satisfaction(candidates, required)
    assert verdict.satisfied is True
    assert len(verdict.violated_constraints) == 0
    assert verdict.details["101"]["status"] == "PASS"


def test_judge_constraint_satisfaction_violations():
    """Verify constraint checker correctly detects multiple filter violations."""
    candidates = [
        {
            "anime_id": 999,
            "season_year": 2012,  # < min 2018
            "format": "MOVIE",    # != TV
            "average_score": 65,  # < min 80
            "genres": "Romance",  # missing Action
        }
    ]
    required = {
        "year_min": 2018,
        "format": "TV",
        "min_score": 80,
        "genres": ["Action"],
    }

    verdict = judge_constraint_satisfaction(candidates, required)
    assert verdict.satisfied is False
    assert len(verdict.violated_constraints) == 4
    assert any("Year 2012 < min 2018" in v for v in verdict.violated_constraints)
    assert any("Format 'MOVIE' != required 'TV'" in v for v in verdict.violated_constraints)
    assert any("Score 65 < min 80" in v for v in verdict.violated_constraints)
    assert any("Missing required genre 'Action'" in v for v in verdict.violated_constraints)


def test_evaluate_generation_batch():
    """Test generation batch summary aggregation."""
    records = [
        {"faithfulness_score": 5, "is_faithful": True, "constraint_satisfied": True},
        {"faithfulness_score": 4, "is_faithful": True, "constraint_satisfied": True},
        {"faithfulness_score": 2, "is_faithful": False, "constraint_satisfied": False},
        {"faithfulness_score": 5, "is_faithful": True, "constraint_satisfied": True},
    ]

    summary = evaluate_generation_batch(records)
    # Avg score: (5 + 4 + 2 + 5) / 4 = 16 / 4 = 4.0
    # Faithful rate: 3 / 4 = 0.75
    # Constraint satisfaction rate: 3 / 4 = 0.75
    assert summary["avg_faithfulness_score"] == 4.0
    assert summary["faithfulness_rate"] == 0.75
    assert summary["constraint_satisfaction_rate"] == 0.75
    assert summary["total_evaluated"] == 4.0


def test_evaluate_generation_batch_empty():
    """Test empty batch handling."""
    summary = evaluate_generation_batch([])
    assert summary["avg_faithfulness_score"] == 0.0
    assert summary["faithfulness_rate"] == 0.0
    assert summary["total_evaluated"] == 0.0
