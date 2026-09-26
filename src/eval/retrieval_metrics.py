"""Statistical retrieval evaluation metrics: Hit Rate @ K, MRR, Precision @ K, and Recall @ K."""

from typing import Any, Dict, List, Sequence, Set
from pydantic import BaseModel, Field


class RetrievalMetricsSummary(BaseModel):
    """Structured summary of retrieval evaluation metrics over a test batch."""

    total_queries: int = Field(..., description="Number of evaluated queries")
    hit_rate_at_k: float = Field(..., description="Proportion of queries with at least one hit in top K")
    mrr: float = Field(..., description="Mean Reciprocal Rank of first relevant hit in top K")
    precision_at_k: float = Field(default=0.0, description="Average precision @ K")
    recall_at_k: float = Field(default=0.0, description="Average recall @ K")
    k: int = Field(default=5, description="Cutoff rank K used for metrics")


def calculate_hit_at_k(
    retrieved_ids: Sequence[int],
    expected_ids: Set[int] | Sequence[int],
    k: int = 5,
) -> float:
    """Calculate Hit@K for a single query.

    Returns 1.0 if at least one expected ID is in the top-k retrieved IDs, else 0.0.
    If expected_ids is empty, returns 1.0 if retrieved_ids is also empty, else 0.0.
    """
    expected_set = set(expected_ids)
    if not expected_set:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_retrieved & expected_set) else 0.0


def calculate_mrr(
    retrieved_ids: Sequence[int],
    expected_ids: Set[int] | Sequence[int],
    k: int = 5,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query within top-k.

    Returns 1 / rank of the first relevant document if within top-k, else 0.0.
    If expected_ids is empty, returns 1.0 if retrieved_ids is empty, else 0.0.
    """
    expected_set = set(expected_ids)
    if not expected_set:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in expected_set:
            return 1.0 / rank
    return 0.0


def calculate_precision_at_k(
    retrieved_ids: Sequence[int],
    expected_ids: Set[int] | Sequence[int],
    k: int = 5,
) -> float:
    """Calculate Precision@K: (number of relevant items in top-k) / k."""
    expected_set = set(expected_ids)
    if not expected_set:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    top_k_retrieved = retrieved_ids[:k]
    if not top_k_retrieved:
        return 0.0

    relevant_count = sum(1 for rid in top_k_retrieved if rid in expected_set)
    return relevant_count / len(top_k_retrieved)


def calculate_recall_at_k(
    retrieved_ids: Sequence[int],
    expected_ids: Set[int] | Sequence[int],
    k: int = 5,
) -> float:
    """Calculate Recall@K: (number of relevant items in top-k) / (total expected items)."""
    expected_set = set(expected_ids)
    if not expected_set:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    top_k_retrieved = retrieved_ids[:k]
    relevant_count = sum(1 for rid in top_k_retrieved if rid in expected_set)
    return relevant_count / len(expected_set)


def evaluate_retrieval_batch(
    results: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, float]:
    """Compute aggregate retrieval metrics over a batch of evaluation records.

    Each item in results is expected to have:
      - 'retrieved_ids' (List[int]) OR 'retrieved_candidates' (List[Dict] with metadata.anilist_id / metadata.anime_id)
      - 'expected_anilist_ids' (List[int]) OR 'expected_ids' (List[int])

    Returns:
        Dict with hit_rate_at_{k}, mrr, precision_at_{k}, recall_at_{k}, and total_queries.
    """
    if not results:
        return {
            f"hit_rate_at_{k}": 0.0,
            "mrr": 0.0,
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            "total_queries": 0.0,
        }

    hits: List[float] = []
    mrrs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for item in results:
        # Extract retrieved IDs
        if "retrieved_ids" in item:
            retrieved = item["retrieved_ids"]
        elif "retrieved_candidates" in item:
            retrieved = []
            for c in item["retrieved_candidates"]:
                if isinstance(c, dict):
                    meta = c.get("metadata", {})
                    aid = meta.get("anilist_id") or meta.get("anime_id") or c.get("anime_id") or c.get("id")
                    if aid is not None:
                        try:
                            retrieved.append(int(aid))
                        except (ValueError, TypeError):
                            pass
                elif hasattr(c, "metadata"):
                    meta = getattr(c, "metadata")
                    aid = meta.get("anilist_id") if isinstance(meta, dict) else getattr(meta, "anilist_id", None)
                    if aid is not None:
                        retrieved.append(int(aid))
        else:
            retrieved = []

        # Extract expected IDs
        expected = item.get("expected_anilist_ids") or item.get("expected_ids") or []

        hits.append(calculate_hit_at_k(retrieved, expected, k=k))
        mrrs.append(calculate_mrr(retrieved, expected, k=k))
        precisions.append(calculate_precision_at_k(retrieved, expected, k=k))
        recalls.append(calculate_recall_at_k(retrieved, expected, k=k))

    total = max(len(results), 1)
    return {
        f"hit_rate_at_{k}": round(sum(hits) / total, 4),
        "mrr": round(sum(mrrs) / total, 4),
        f"precision_at_{k}": round(sum(precisions) / total, 4),
        f"recall_at_{k}": round(sum(recalls) / total, 4),
        "total_queries": float(len(results)),
    }
