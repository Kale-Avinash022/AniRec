# Step 50: Retrieval Metrics Implementation (Hit Rate @ 5, MRR)

## 1. Process & Purpose
Implement objective statistical metrics to evaluate retrieval quality over the 50-pair golden benchmark: **Hit Rate @ K** (target $\ge 0.70$) and **Mean Reciprocal Rank (MRR)**.

---

## 2. Mathematical Formulations

### Hit Rate @ K
$$\text{Hit@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{Any}(e \in \text{Expected}(q) \text{ in } \text{TopK}(q)) \right)$$

### Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{Rank of first relevant show in TopK}}$$

---

## 3. Implementation Code

```python
# evals/retrieval_metrics.py
from typing import List, Set, Dict, Any

def calculate_hit_at_k(retrieved_ids: List[int], expected_ids: Set[int], k: int = 5) -> float:
    top_k_retrieved = set(retrieved_ids[:k])
    return 1.0 if bool(top_k_retrieved & expected_ids) else 0.0

def calculate_mrr(retrieved_ids: List[int], expected_ids: Set[int], k: int = 5) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0

def evaluate_retrieval_batch(results: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    hits = []
    mrrs = []
    
    for item in results:
        retrieved = [c["metadata"]["anilist_id"] for c in item["retrieved_candidates"]]
        expected = set(item["expected_anilist_ids"])
        
        hits.append(calculate_hit_at_k(retrieved, expected, k=k))
        mrrs.append(calculate_mrr(retrieved, expected, k=k))

    return {
        f"hit_rate_at_{k}": round(sum(hits) / max(len(hits), 1), 4),
        "mrr": round(sum(mrrs) / max(len(mrrs), 1), 4)
    }
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Hit Rate @ 5 and MRR calculation functions implemented with unit tests.
- [x] Target threshold benchmark: Hit Rate @ 5 $\ge 0.70$.
