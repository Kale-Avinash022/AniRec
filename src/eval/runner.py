"""Automated Evaluation Suite Runner for AniRec (Step 52).

Executes the end-to-end multi-stage pipeline across the golden dataset:
1. Query Understanding & Filter Extraction
2. Chroma Vector Search with Progressive Filter Relaxation
3. Cross-Encoder Reranking
4. Franchise Deduplication & Diversity Filtering
5. Grounded Recommendation Generation
6. Retrieval Metrics Evaluation (Hit@5, MRR)
7. LLM Judge Faithfulness & Constraint Satisfaction Evaluation
8. Latency Profiling & Timestamped Run Report Persistence
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.interfaces import BaseChatModel
from src.core.logger import logger
from src.eval.dataset import GoldenQueryItem, load_golden_set
from src.eval.judge import judge_constraint_satisfaction, judge_faithfulness
from src.eval.retrieval_metrics import calculate_hit_at_k, calculate_mrr
from src.generation.llm_client import get_chat_model
from src.generation.pipeline import GroundedRecommender
from src.retrieval.dedup import deduplicate_franchises
from src.retrieval.embedder import get_embedder
from src.retrieval.query_parser import parse_query_rules
from src.retrieval.relaxation import search_with_relaxation
from src.retrieval.reranker import Reranker
from src.retrieval.vectorstore import AnimeVectorStore


async def run_evaluation_suite(
    experiment_name: str = "production_pipeline",
    max_samples: Optional[int] = None,
    golden_path: Optional[str | Path] = None,
    out_dir: Optional[str | Path] = None,
    llm: Optional[BaseChatModel] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the end-to-end evaluation harness across the benchmark suite.

    Args:
        experiment_name: Name identifier for this benchmark run.
        max_samples: Optional cap on the number of evaluation items to run.
        golden_path: Custom path to golden evaluation dataset JSON.
        out_dir: Destination folder for output run JSON reports.
        llm: Custom chat model instance (e.g. MockChatModel for tests).
        verbose: If True, logs progress and summary to stdout.

    Returns:
        Dict containing "summary" and "details" records.
    """
    golden_data = load_golden_set(golden_path)
    if max_samples is not None and max_samples > 0:
        golden_data = golden_data[:max_samples]

    store = AnimeVectorStore()
    embedder = get_embedder()
    reranker = Reranker()
    if llm is None:
        llm = get_chat_model()
    recommender = GroundedRecommender(llm=llm)

    eval_records: List[Dict[str, Any]] = []
    latencies: List[float] = []

    if verbose:
        print(f"🚀 Running Evaluation Experiment: '{experiment_name}' on {len(golden_data)} items...")

    for idx, item in enumerate(golden_data, start=1):
        start_t = time.perf_counter()
        query = item.query
        expected_ids = set(item.expected_anilist_ids)

        # 1. Parse query constraints & semantic text
        parsed = parse_query_rules(query)

        # 2. Vector Search with Progressive Filter Relaxation
        candidates, relaxation_notes = search_with_relaxation(
            store=store,
            embedder=embedder,
            semantic_text=parsed.semantic_text,
            initial_filters=parsed.filters,
            top_k=20,
        )

        # 3. Cross-Encoder Reranking
        rescored = reranker.rerank(query=parsed.semantic_text, candidates=candidates, top_k=10)

        # 4. Franchise Deduplication
        deduped = deduplicate_franchises(rescored, max_per_franchise=1)[:5]

        # 5. Grounded Recommendation Generation
        rec_output = await recommender.recommend(user_query=query, candidates=deduped)
        duration_ms = (time.perf_counter() - start_t) * 1000.0

        latencies.append(duration_ms)

        # 6. Retrieval Metrics (Hit@5, MRR)
        retrieved_ids = [
            int(c["metadata"].get("anime_id") or c["metadata"].get("anilist_id"))
            for c in deduped
            if "metadata" in c and (c["metadata"].get("anime_id") or c["metadata"].get("anilist_id"))
        ]
        hit_at_5 = calculate_hit_at_k(retrieved_ids, expected_ids, k=5)
        mrr = calculate_mrr(retrieved_ids, expected_ids, k=5)

        # 7. Constraint Satisfaction Check
        candidate_metas = [c.get("metadata", {}) for c in deduped]
        constraint_verdict = judge_constraint_satisfaction(candidate_metas, item.required_filters)

        # 8. LLM Judge Faithfulness
        context_str = "\n\n".join([
            f"Title: {c.get('metadata', {}).get('title_english') or c.get('metadata', {}).get('title_romaji')}\n"
            f"Genres: {c.get('metadata', {}).get('genres')}\n"
            f"Synopsis: {c.get('document', '')}"
            for c in deduped
        ])
        answer_str = "\n".join([
            f"Show: {rec.title} (ID: {rec.anilist_id})\nWhy: {rec.why}\nAttributes: {', '.join(rec.matched_attributes)}"
            for rec in rec_output.recommendations
        ])
        judge_verdict = await judge_faithfulness(context=context_str, answer=answer_str, llm=llm)


        record = {
            "id": item.id,
            "category": item.category,
            "query": query,
            "expected_ids": list(expected_ids),
            "retrieved_ids": retrieved_ids,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
            "faithfulness_score": judge_verdict.score,
            "is_faithful": judge_verdict.is_faithful,
            "judge_reasoning": judge_verdict.reasoning,
            "constraint_satisfied": constraint_verdict.satisfied,
            "violated_constraints": constraint_verdict.violated_constraints,
            "recommendation_count": len(rec_output.recommendations),
            "relaxation_notes": relaxation_notes,
            "latency_ms": round(duration_ms, 2),
        }
        eval_records.append(record)

        if verbose:
            print(
                f"[{idx:02d}/{len(golden_data):02d}] {item.id} | "
                f"Hit@5: {hit_at_5:.1f} | MRR: {mrr:.2f} | "
                f"Faithful: {judge_verdict.score}/5 ({'PASS' if judge_verdict.is_faithful else 'FAIL'}) | "
                f"Latency: {duration_ms:.0f}ms"
            )

    # Aggregate Statistics
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
        "experiment_name": experiment_name,
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

    # Save to evals/results/
    target_out_dir = Path(out_dir) if out_dir else Path("evals/results")
    target_out_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"eval_{experiment_name}_{ts_str}.json"
    report_path = target_out_dir / report_filename

    full_report = {
        "summary": summary,
        "details": eval_records,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 60)
        print("📊 EVALUATION RUN COMPLETED")
        print("=" * 60)
        print(f"Report: {report_path}")
        print(f"Total Queries Tested: {summary['total_queries']}")
        print(f"Hit Rate @ 5:         {summary['hit_rate_at_5'] * 100:.1f}%")
        print(f"MRR:                  {summary['mrr']:.4f}")
        print(f"Faithfulness Rate:    {summary['faithfulness_rate'] * 100:.1f}% (Avg Score: {summary['avg_faithfulness_score']}/5)")
        print(f"Constraint Pass Rate: {summary['constraint_satisfaction_rate'] * 100:.1f}%")
        print(f"p50 Latency:          {summary['p50_latency_ms']} ms")
        print(f"p95 Latency:          {summary['p95_latency_ms']} ms")
        print("=" * 60 + "\n")

    return full_report
