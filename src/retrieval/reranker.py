"""Cross-Encoder Reranker Integration (Step 29).

Two-stage retrieval architecture:
  Stage 1 — Bi-encoder (ChromaDB)  : fast approximate nearest-neighbour search
             Retrieves top-20 candidates in ~5ms
             Encodes query and documents SEPARATELY → misses token interactions

  Stage 2 — Cross-encoder           : precise relevance scoring
             Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
             Feeds (query, document) JOINTLY through transformer self-attention
             Re-scores all 20 candidates → returns top-K by cross-attention logit

Why this matters (the "dark political fantasy" example from verification):
  Bi-encoder scored Cowboy Bebop at 0.566 for "space bounty hunters" — good.
  But for "dark political fantasy", bi-encoder returned unrelated titles at 0.41–0.45.
  Cross-encoder sees *both* query tokens and document tokens simultaneously, so
  "military strategy" in the document matches "political fantasy" in the query
  via cross-attention — yielding much better precision.

Model details:
  - ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  - Trained on MS MARCO passage ranking
  - Returns raw logit (not bounded) — higher = more relevant
  - CPU latency: ~80–120ms for 20 pairs
  - Downloaded once, cached in ~/.cache/huggingface/
"""

from __future__ import annotations

from typing import Any

from src.core.config import settings
from src.core.logger import logger


class Reranker:
    """Cross-Encoder reranker using ``ms-marco-MiniLM-L-6-v2``.

    Scores (query, document_text) pairs and re-orders candidates by
    cross-attention relevance logit.

    Args:
        model_name: HuggingFace model ID (default from ``settings.CROSS_ENCODER_MODEL``).
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        _model = model_name or settings.CROSS_ENCODER_MODEL
        logger.info(f"Loading Cross-Encoder model: {_model}")
        self.model = CrossEncoder(_model)
        self.model_name = _model
        logger.info("Cross-Encoder ready")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score and re-order candidates by cross-attention relevance.

        Args:
            query:      The original user query string (raw, not semantic_text).
            candidates: List of candidate dicts from :func:`~src.retrieval.search.search_candidates`.
                        Each must have a ``"document"`` key containing the full document text.
            top_k:      Return only the top-K results after reranking.
                        Defaults to ``settings.TOP_K_RERANKED`` (5).

        Returns:
            Candidates sorted descending by ``rerank_score``, truncated to ``top_k``.
        """
        if not candidates:
            return []

        _top_k = top_k if top_k is not None else settings.TOP_K_RERANKED

        # Cap candidates to rerank pool for sub-second transformer inference
        pool_size = max(_top_k + 5, 20)
        eval_pool = candidates[:pool_size]

        # Build (query, document) pairs; truncate text to 350 chars for fast transformer attention
        pairs = [[query, c.get("document", "")[:350]] for c in eval_pool]

        # Predict relevance logits with optimal batching
        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        # Attach rerank_score to each candidate
        reranked = []
        for candidate, score in zip(eval_pool, scores):
            enriched = dict(candidate)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        # If candidates exceeded pool_size, append remaining with fallback score
        if len(candidates) > pool_size:
            for remaining in candidates[pool_size:]:
                enriched = dict(remaining)
                enriched["rerank_score"] = float(remaining.get("similarity", 0.0)) - 1.0
                reranked.append(enriched)

        # Sort descending by cross-encoder logit
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            f"Reranker: {len(candidates)} → {_top_k} | "
            f"top score={reranked[0]['rerank_score']:.3f} | "
            f"top title={reranked[0]['metadata'].get('title_romaji', '?')}"
        )

        return reranked[:_top_k]
