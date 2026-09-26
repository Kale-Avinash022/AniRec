"""Ablation Benchmarking & Iterative Optimization Experiments for AniRec (Step 54).

Provides ablation runner modes to quantitatively measure the impact of each subsystem:
1. `baseline_vector_only`: Naive Chroma similarity search without reranker or franchise dedup.
2. `with_cross_encoder`: Chroma search + Cross-Encoder (ms-marco-MiniLM-L-6-v2) reranking.
3. `with_franchise_dedup`: Chroma + Cross-Encoder + Franchise graph deduplication.
4. `full_production_guardrails`: Full pipeline with Progressive Relaxation + Anti-hallucination verification gate.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.core.interfaces import BaseChatModel
from src.core.logger import logger
from src.eval.dataset import load_golden_set
from src.eval.judge import judge_constraint_satisfaction, judge_faithfulness
from src.eval.retrieval_metrics import calculate_hit_at_k, calculate_mrr
from src.generation.llm_client import get_chat_model
from src.generation.pipeline import GroundedRecommender, build_deterministic_fallback
from src.retrieval.dedup import deduplicate_franchises
from src.retrieval.embedder import get_embedder
from src.retrieval.query_parser import parse_query_rules
from src.retrieval.relaxation import search_with_relaxation
from src.retrieval.reranker import Reranker
from src.retrieval.search import search_candidates
from src.retrieval.vectorstore import AnimeVectorStore


AblationMode = Literal[
    "01_baseline_vector_only",
    "02_with_cross_encoder",
    "03_with_franchise_dedup",
    "04_full_production_guardrails",
]


async def run_ablation_experiment(
    mode: AblationMode,
    max_samples: Optional[int] = None,
    golden_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    llm: Optional[BaseChatModel] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a single ablation configuration across the golden dataset.

    Args:
        mode: Ablation configuration mode.
        max_samples: Optional limit on test query count.
        golden_path: Custom path to golden_set.json.
        out_dir: Output directory for benchmark run JSONs.
        llm: Optional LLM model adapter.
        verbose: Verbose logging to console.

    Returns:
        Dict with "summary" and "details".
    """
    golden_data = load_golden_set(golden_path)
    if max_samples is not None and max_samples > 0:
        golden_data = golden_data[:max_samples]

    store = AnimeVectorStore()
    embedder = get_embedder()
    reranker = Reranker() if mode != "01_baseline_vector_only" else None
    if llm is None:
        llm = get_chat_model()
    recommender = GroundedRecommender(llm=llm)

    eval_records: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for item in golden_data:
        start_t = time.perf_counter()
        query = item.query
        expected_ids = set(item.expected_anilist_ids)
        parsed = parse_query_rules(query)

        # ── Step 1: Retrieval Phase based on Ablation Mode ───────────────────
        if mode == "01_baseline_vector_only":
            # Naive unrelaxed vector search, no cross-encoder, no dedup
            raw_candidates = search_candidates(
                store=store,
                embedder=embedder,
                semantic_text=parsed.semantic_text,
                filters=parsed.filters,
                top_k=5,
            )
            final_candidates = raw_candidates[:5]
            relaxation_notes = []

        elif mode == "02_with_cross_encoder":
            # Chroma vector search + Cross-Encoder reranking (no dedup)
            raw_candidates = search_candidates(
                store=store,
                embedder=embedder,
                semantic_text=parsed.semantic_text,
                filters=parsed.filters,
                top_k=20,
            )
            assert reranker is not None
            rescored = reranker.rerank(query=parsed.semantic_text, candidates=raw_candidates, top_k=5)
            final_candidates = rescored[:5]
            relaxation_notes = []

        elif mode == "03_with_franchise_dedup":
            # Chroma vector search + Cross-Encoder reranking + Franchise Dedup
            raw_candidates = search_candidates(
                store=store,
                embedder=embedder,
                semantic_text=parsed.semantic_text,
                filters=parsed.filters,
                top_k=20,
            )
            assert reranker is not None
            rescored = reranker.rerank(query=parsed.semantic_text, candidates=raw_candidates, top_k=10)
            deduped = deduplicate_franchises(rescored, max_per_franchise=1)
            final_candidates = deduped[:5]
            relaxation_notes = []

        elif mode == "04_full_production_guardrails":
            # Full progressive relaxation + Cross-Encoder + Dedup + Guardrails
            candidates, relaxation_notes = search_with_relaxation(
                store=store,
                embedder=embedder,
                semantic_text=parsed.semantic_text,
                initial_filters=parsed.filters,
                top_k=20,
            )
            assert reranker is not None
            rescored = reranker.rerank(query=parsed.semantic_text, candidates=candidates, top_k=10)
            deduped = deduplicate_franchises(rescored, max_per_franchise=1)
            final_candidates = deduped[:5]

        # ── Step 2: Generation Phase ─────────────────────────────────────────
        if mode == "01_baseline_vector_only":
            # Baseline uses deterministic fallback template
            rec_output = build_deterministic_fallback(user_query=query, candidates=final_candidates, reason="Baseline")
        else:
            rec_output = await recommender.recommend(user_query=query, candidates=final_candidates)

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(duration_ms)

        # ── Step 3: Metrics Evaluation ───────────────────────────────────────
        retrieved_ids = [
            int(c["metadata"].get("anime_id") or c["metadata"].get("anilist_id"))
            for c in final_candidates
            if "metadata" in c and (c["metadata"].get("anime_id") or c["metadata"].get("anilist_id"))
        ]
        hit_at_5 = calculate_hit_at_k(retrieved_ids, expected_ids, k=5)
        mrr = calculate_mrr(retrieved_ids, expected_ids, k=5)

        candidate_metas = [c.get("metadata", {}) for c in final_candidates]
        constraint_verdict = judge_constraint_satisfaction(candidate_metas, item.required_filters)

        context_str = "\n\n".join([
            f"Title: {c.get('metadata', {}).get('title_english') or c.get('metadata', {}).get('title_romaji')}\n"
            f"Genres: {c.get('metadata', {}).get('genres')}\n"
            f"Synopsis: {c.get('document', '')}"
            for c in final_candidates
        ])
        answer_str = "\n".join([
            f"Show: {rec.title} (ID: {rec.anilist_id})\nWhy: {rec.why}\nAttributes: {', '.join(rec.matched_attributes)}"
            for rec in rec_output.recommendations
        ])
        judge_verdict = await judge_faithfulness(context=context_str, answer=answer_str, llm=llm)

        eval_records.append({
            "id": item.id,
            "category": item.category,
            "query": query,
            "expected_ids": list(expected_ids),
            "retrieved_ids": retrieved_ids,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
            "faithfulness_score": judge_verdict.score,
            "is_faithful": judge_verdict.is_faithful,
            "constraint_satisfied": constraint_verdict.satisfied,
            "latency_ms": round(duration_ms, 2),
        })

    total_q = max(len(eval_records), 1)
    avg_hit = sum(r["hit_at_5"] for r in eval_records) / total_q
    avg_mrr = sum(r["mrr"] for r in eval_records) / total_q
    avg_faith_score = sum(r["faithfulness_score"] for r in eval_records) / total_q
    faith_rate = sum(1 for r in eval_records if r["is_faithful"]) / total_q
    constraint_rate = sum(1 for r in eval_records if r["constraint_satisfied"]) / total_q

    sorted_lats = sorted(latencies) if latencies else [0.0]
    p50_lat = sorted_lats[int(len(sorted_lats) * 0.50)]
    p95_lat = sorted_lats[min(int(len(sorted_lats) * 0.95), len(sorted_lats) - 1)]

    summary = {
        "experiment_name": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(eval_records),
        "hit_rate_at_5": round(avg_hit, 4),
        "mrr": round(avg_mrr, 4),
        "avg_faithfulness_score": round(avg_faith_score, 2),
        "faithfulness_rate": round(faith_rate, 4),
        "constraint_satisfaction_rate": round(constraint_rate, 4),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
    }

    target_out_dir = Path(out_dir) if out_dir else Path("evals/results")
    target_out_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"eval_{mode}.json"
    report_path = target_out_dir / report_filename

    full_report = {
        "summary": summary,
        "details": eval_records,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✅ Ablation '{mode}' completed -> Hit@5: {summary['hit_rate_at_5'] * 100:.1f}%, MRR: {summary['mrr']:.3f}, Faithfulness: {summary['faithfulness_rate'] * 100:.1f}%")

    return full_report


async def run_all_ablations(
    max_samples: Optional[int] = None,
    golden_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    llm: Optional[BaseChatModel] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute all four ablation experiments in sequence and return the comparative report.

    Args:
        max_samples: Optional cap on queries.
        golden_path: Custom path to golden dataset.
        out_dir: Target results directory.
        llm: Custom LLM provider.
        verbose: Print progress.

    Returns:
        Dict mapping ablation mode to report.
    """
    modes: List[AblationMode] = [
        "01_baseline_vector_only",
        "02_with_cross_encoder",
        "03_with_franchise_dedup",
        "04_full_production_guardrails",
    ]

    all_reports = {}
    for mode in modes:
        if verbose:
            print(f"🔬 Starting Ablation Benchmark: {mode}...")
        report = await run_ablation_experiment(
            mode=mode,
            max_samples=max_samples,
            golden_path=golden_path,
            out_dir=out_dir,
            llm=llm,
            verbose=verbose,
        )
        all_reports[mode] = report

    return all_reports
