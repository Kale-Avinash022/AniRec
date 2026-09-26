"""Tests for Maximal Marginal Relevance (MMR) & Diversity Filtering (Step 31)."""

import pytest
import numpy as np
from src.retrieval.diversity import cosine_sim, mmr_diversity_filter


def test_cosine_sim():
    # Identical vectors -> 1.0
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_sim(v1, v2), 1e-5) == 1.0

    # Orthogonal vectors -> 0.0
    v3 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(cosine_sim(v1, v3), 1e-5) == 0.0

    # Zero vector -> 0.0 (safe against Div0)
    v_zero = np.array([0.0, 0.0, 0.0])
    assert cosine_sim(v1, v_zero) == 0.0


def test_mmr_candidates_less_than_top_k():
    candidates = [{"id": "1"}, {"id": "2"}]
    res = mmr_diversity_filter(
        query_vector=[1.0, 0.0],
        candidates=candidates,
        candidate_vectors=[[1.0, 0.0], [0.0, 1.0]],
        top_k=5,
    )
    assert len(res) == 2


def test_mmr_diversity_penalizes_redundant_duplicates():
    # Query vector at 45 degrees between axis-0 and axis-1
    q_vec = [1.0, 1.0, 0.0]

    # Candidate 0: aligned with axis-0
    # Candidate 1: almost identical to Candidate 0 (high redundancy with c0)
    # Candidate 2: aligned with axis-1 (orthogonal to c0, equally relevant to query)
    c0 = [1.0, 0.0, 0.0]
    c1 = [0.999, 0.001, 0.0]
    c2 = [0.0, 1.0, 0.0]

    candidates = [
        {"id": "c0", "name": "Main Target"},
        {"id": "c1", "name": "Near Clone of c0"},
        {"id": "c2", "name": "Diverse Alternative"},
    ]
    cand_vecs = [c0, c1, c2]

    # With lambda_param=0.5 (balanced), after picking c0, c1 has high redundancy (1.0) with c0
    # while c2 has zero redundancy (0.0) with c0.
    # Therefore c2 has higher MMR score and is chosen for slot 2.
    selected = mmr_diversity_filter(
        query_vector=q_vec,
        candidates=candidates,
        candidate_vectors=cand_vecs,
        top_k=2,
        lambda_param=0.5,
    )

    assert len(selected) == 2
    selected_ids = [c["id"] for c in selected]
    # c2 must be chosen (diversity over the redundant clone)
    assert "c2" in selected_ids
    # Only one of c0 or c1 is chosen, not both
    assert ("c0" in selected_ids and "c1" not in selected_ids) or ("c1" in selected_ids and "c0" not in selected_ids)
