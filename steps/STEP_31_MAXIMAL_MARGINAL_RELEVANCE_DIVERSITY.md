# Step 31: Maximal Marginal Relevance (MMR) & Diversity Filtering

## 1. Process & Purpose
Ensure the final Top-5 recommendations provide a diverse mix of themes, tones, and tags rather than 5 nearly identical shows (e.g. 5 generic isekai series). Maximal Marginal Relevance (MMR) balances query relevance against pairwise document redundancy.

---

## 2. MMR Mathematical Formula

$$\text{MMR} = \operatorname{argmax}_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$

- $\lambda = 0.7$: Prioritizes query relevance while penalizing similarity to already selected shows.
- $R$: Candidate pool.
- $S$: Currently selected output set.

---

## 3. Implementation Code

```python
# src/retrieval/diversity.py
import numpy as np
from typing import List, Dict, Any

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def mmr_diversity_filter(
    query_vector: List[float],
    candidates: List[Dict[str, Any]],
    candidate_vectors: List[List[float]],
    top_k: int = 5,
    lambda_param: float = 0.7
) -> List[Dict[str, Any]]:
    if len(candidates) <= top_k:
        return candidates

    q_vec = np.array(query_vector)
    cand_vecs = [np.array(v) for v in candidate_vectors]
    
    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidates)))

    # Step 1: Select show with highest query similarity
    query_sims = [cosine_sim(q_vec, cv) for cv in cand_vecs]
    best_initial = int(np.argmax(query_sims))
    selected_indices.append(best_initial)
    unselected_indices.remove(best_initial)

    # Step 2: Iteratively select remaining shows balancing relevance & novelty
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx = None

        for idx in unselected_indices:
            relevance = query_sims[idx]
            # Max similarity to already selected candidates
            redundancy = max([cosine_sim(cand_vecs[idx], cand_vecs[s]) for s in selected_indices])
            mmr_val = (lambda_param * relevance) - ((1.0 - lambda_param) * redundancy)

            if mmr_val > best_score:
                best_score = mmr_val
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]
```

---

## 4. Deliverables & Acceptance Criteria
- [x] Vector-based MMR diversity calculation implemented.
- [x] Configurable $\lambda$ parameter tuning the relevance vs diversity balance.
