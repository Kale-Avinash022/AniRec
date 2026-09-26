"""Maximal Marginal Relevance (MMR) & Diversity Filtering (Step 31).

Ensures the final Top-5 recommendation set provides a diverse mix of themes,
tones, and subgenres rather than multiple nearly identical shows.

MMR balances relevance against pairwise redundancy:
    MMR = argmax_{d_i in R \\ S} [ lambda * Sim(d_i, q) - (1 - lambda) * max_{d_j in S} Sim(d_i, d_j) ]

Where:
    - lambda in [0, 1] (default 0.7): higher lambda emphasizes query relevance,
      lower lambda promotes novelty/diversity.
    - R: candidate pool (from reranker / dedup).
    - S: currently selected output set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.config import settings
from src.core.logger import logger


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D numerical vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr_diversity_filter(
    query_vector: list[float] | np.ndarray,
    candidates: list[dict[str, Any]],
    candidate_vectors: list[list[float]] | list[np.ndarray],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """Select top_k diverse candidates using Maximal Marginal Relevance.

    Args:
        query_vector:      Embedding vector of the search query.
        candidates:        List of candidate dicts (e.g. from dedup / reranker).
        candidate_vectors: List of embedding vectors corresponding 1-to-1 with candidates.
        top_k:             Number of diverse items to select (default: 5).
        lambda_param:      Balance factor between relevance (1.0) and diversity (0.0).

    Returns:
        Ordered list of top_k diverse candidate dicts.
    """
    if len(candidates) <= top_k or not candidates:
        return candidates

    if len(candidates) != len(candidate_vectors):
        logger.warning(
            f"Candidate count ({len(candidates)}) mismatch with vector count ({len(candidate_vectors)}). Returning top_k."
        )
        return candidates[:top_k]

    q_vec = np.asarray(query_vector, dtype=np.float32)
    cand_vecs = [np.asarray(v, dtype=np.float32) for v in candidate_vectors]

    selected_indices: list[int] = []
    unselected_indices: list[int] = list(range(len(candidates)))

    # Compute query similarities for all candidates
    query_sims = [cosine_sim(q_vec, cv) for cv in cand_vecs]

    # Step 1: Select the candidate with highest query similarity
    best_initial = int(np.argmax(query_sims))
    selected_indices.append(best_initial)
    unselected_indices.remove(best_initial)

    # Step 2: Greedily pick remaining items maximizing MMR score
    while len(selected_indices) < top_k and unselected_indices:
        best_score = -float("inf")
        best_idx: int | None = None

        for idx in unselected_indices:
            relevance = query_sims[idx]
            # Max similarity to any already selected candidate
            redundancy = max(cosine_sim(cand_vecs[idx], cand_vecs[s]) for s in selected_indices)
            mmr_val = (lambda_param * relevance) - ((1.0 - lambda_param) * redundancy)

            if mmr_val > best_score:
                best_score = mmr_val
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)

    selected_candidates = [candidates[i] for i in selected_indices]
    logger.debug(
        f"MMR diversity filter: selected {len(selected_candidates)} from {len(candidates)} (lambda={lambda_param})"
    )
    return selected_candidates
