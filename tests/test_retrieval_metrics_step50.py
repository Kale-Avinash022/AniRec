"""Unit tests for Step 50: Retrieval Metrics Implementation (Hit Rate @ K, MRR)."""

import pytest
from src.eval.retrieval_metrics import (
    RetrievalMetricsSummary,
    calculate_hit_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval_batch,
)
from src.eval.dataset import load_golden_set
from src.retrieval.vectorstore import AnimeVectorStore
from src.retrieval.embedder import get_embedder
from src.retrieval.search import search_candidates
from src.retrieval.relaxation import search_with_relaxation
from src.retrieval.reranker import Reranker
from src.retrieval.dedup import deduplicate_franchises
from src.retrieval.query_parser import parse_query_rules




def test_calculate_hit_at_k():
    """Test Hit@K calculation across diverse ranking positions."""
    expected = {101, 102}

    # Hit at rank 1
    assert calculate_hit_at_k([101, 200, 300], expected, k=5) == 1.0

    # Hit at rank 4 (within k=5)
    assert calculate_hit_at_k([200, 300, 400, 102, 500], expected, k=5) == 1.0

    # Hit at rank 6 (outside k=5)
    assert calculate_hit_at_k([200, 300, 400, 500, 600, 101], expected, k=5) == 0.0

    # No overlap
    assert calculate_hit_at_k([1, 2, 3], expected, k=5) == 0.0

    # Empty retrieved vs empty expected
    assert calculate_hit_at_k([], set(), k=5) == 1.0
    assert calculate_hit_at_k([1, 2], set(), k=5) == 0.0


def test_calculate_mrr():
    """Test Mean Reciprocal Rank (MRR) mathematical computation."""
    expected = {100, 200}

    # First hit at rank 1 -> 1.0
    assert calculate_mrr([100, 200, 300], expected, k=5) == 1.0

    # First hit at rank 2 -> 0.5
    assert calculate_mrr([999, 100, 200], expected, k=5) == 0.5

    # First hit at rank 4 -> 0.25
    assert calculate_mrr([1, 2, 3, 200, 5], expected, k=5) == 0.25

    # Hit outside top-k -> 0.0
    assert calculate_mrr([1, 2, 3, 4, 5, 100], expected, k=5) == 0.0

    # No hits -> 0.0
    assert calculate_mrr([1, 2, 3, 4, 5], expected, k=5) == 0.0


def test_precision_and_recall_at_k():
    """Test Precision@K and Recall@K."""
    expected = {10, 20, 30, 40}  # 4 expected items
    retrieved = [10, 20, 99, 88, 77]  # 2 relevant out of top 5

    assert calculate_precision_at_k(retrieved, expected, k=5) == 2 / 5
    assert calculate_recall_at_k(retrieved, expected, k=5) == 2 / 4


def test_evaluate_retrieval_batch():
    """Test batch aggregation over diverse candidate formats."""
    batch = [
        {
            "retrieved_ids": [10, 20, 30],
            "expected_ids": [10],
        },
        {
            "retrieved_ids": [99, 20, 30],
            "expected_ids": [20],
        },
        {
            "retrieved_ids": [99, 88, 77],
            "expected_ids": [10],
        },
        {
            "retrieved_candidates": [
                {"metadata": {"anilist_id": 55}},
                {"metadata": {"anilist_id": 66}},
            ],
            "expected_anilist_ids": [55],
        },
    ]

    metrics = evaluate_retrieval_batch(batch, k=3)
    # Query 1: Hit=1.0, MRR=1.0
    # Query 2: Hit=1.0, MRR=0.5
    # Query 3: Hit=0.0, MRR=0.0
    # Query 4: Hit=1.0, MRR=1.0
    # Avg Hit = 3/4 = 0.75
    # Avg MRR = (1.0 + 0.5 + 0.0 + 1.0) / 4 = 2.5 / 4 = 0.625
    assert metrics["hit_rate_at_3"] == 0.75
    assert metrics["mrr"] == 0.625
    assert metrics["total_queries"] == 4.0


def test_empty_batch_handling():
    """Verify empty results do not raise ZeroDivisionError."""
    metrics = evaluate_retrieval_batch([], k=5)
    assert metrics["hit_rate_at_5"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["total_queries"] == 0.0


def test_real_pipeline_retrieval_on_golden_samples():
    """Test real retrieval pipeline over a sample of golden dataset queries."""
    golden_items = load_golden_set()
    # Test on a representative query with strong keyword and semantic alignment
    test_queries = [it for it in golden_items if it.id in {"eval_02", "eval_07", "eval_18", "eval_21", "eval_40"}]

    store = AnimeVectorStore()
    embedder = get_embedder()
    reranker = Reranker()

    batch_eval_items = []
    for item in test_queries:
        parsed = parse_query_rules(item.query)
        candidates, _ = search_with_relaxation(
            store=store,
            embedder=embedder,
            semantic_text=parsed.semantic_text,
            initial_filters=parsed.filters,
            top_k=20,
        )
        rescored = reranker.rerank(query=parsed.semantic_text, candidates=candidates, top_k=10)
        deduped = deduplicate_franchises(rescored, max_per_franchise=1)

        batch_eval_items.append({
            "retrieved_ids": [c["metadata"]["anime_id"] for c in deduped],
            "expected_anilist_ids": item.expected_anilist_ids,
        })

    metrics = evaluate_retrieval_batch(batch_eval_items, k=5)
    assert "hit_rate_at_5" in metrics
    assert "mrr" in metrics
    assert metrics["total_queries"] == len(test_queries)
    assert metrics["hit_rate_at_5"] > 0.0
    assert metrics["mrr"] > 0.0


