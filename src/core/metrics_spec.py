"""
Project Core Metrics and Target SLO Specifications (Step 01).
Defines quantitative acceptance targets and verification assertions for AniRec.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSLO:
    """Project Success Targets and Service Level Objectives."""

    min_hit_rate_at_5: float = 0.70  # Hit Rate @ 5 >= 70%
    min_faithfulness_score: float = 0.85  # LLM-as-judge Faithfulness >= 85%
    max_p95_latency_seconds: float = 6.0  # p95 end-to-end latency < 6.0s
    max_ttft_seconds: float = 1.5  # Time-To-First-Token streaming < 1.5s
    max_hallucination_rate: float = 0.0  # 0% unverified titles in benchmark
    max_cost_per_query_usd: float = 0.005  # Cost per request < $0.005


@dataclass(frozen=True)
class SkillCompetency:
    component: str
    demonstrated_skill: str
    anti_pattern_avoided: str


SKILL_MAPPING = [
    SkillCompetency(
        component="AniList Ingestion Pipeline",
        demonstrated_skill="Production API engineering, rate-limiting, paginated crawling, snapshot persistence",
        anti_pattern_avoided="Fragile one-off scripts that crash or trigger 429 rate limits",
    ),
    SkillCompetency(
        component="Document Design & Chunking",
        demonstrated_skill="Domain-informed document synthesis, weighted tag ranking",
        anti_pattern_avoided="Naive fixed-size token splitting that fragments context",
    ),
    SkillCompetency(
        component="Hybrid Retrieval",
        demonstrated_skill="Multi-stage retrieval combining semantic embeddings with structured metadata filtering",
        anti_pattern_avoided="Pure vector similarity without constraint awareness",
    ),
    SkillCompetency(
        component="Reranking & Diversity (MMR)",
        demonstrated_skill="Two-stage retrieval, Cross-Encoder scoring, Maximal Marginal Relevance, franchise deduplication",
        anti_pattern_avoided="Top-K similarity clustering of duplicate seasons/spin-offs",
    ),
    SkillCompetency(
        component="Grounded Generation & Guardrails",
        demonstrated_skill="Citation-based prompting, strict JSON contracts, post-generation hallucination verification",
        anti_pattern_avoided="Unconstrained LLM generation with fabricated anime titles",
    ),
    SkillCompetency(
        component="Evals Harness & LLM-as-Judge",
        demonstrated_skill="Rigorous evaluation methodology, Golden Q&A benchmark, quantitative regression tracking",
        anti_pattern_avoided="Subjective vibe testing without measurable benchmarks",
    ),
    SkillCompetency(
        component="FastAPI + SSE Streaming",
        demonstrated_skill="Asynchronous API design, Server-Sent Events, Pydantic contracts",
        anti_pattern_avoided="Blocking synchronous REST endpoints with high perceived latency",
    ),
    SkillCompetency(
        component="Docker & Cloud Deployment",
        demonstrated_skill="Containerization, environment encapsulation, 12-factor cloud deployment",
        anti_pattern_avoided="Local-only uncontainerized prototypes",
    ),
]


def validate_evaluation_metrics(
    metrics: dict[str, float],
    slo: TargetSLO | None = None,
) -> tuple[bool, dict[str, str]]:
    """
    Validates experimental or production metrics against the target project SLOs.
    Returns (passed: bool, report: dict[metric_name, status_message]).
    """
    if slo is None:
        slo = TargetSLO()

    report = {}
    passed = True

    # 1. Hit Rate @ 5
    hit_rate = metrics.get("hit_rate_at_5", 0.0)
    if hit_rate >= slo.min_hit_rate_at_5:
        report["hit_rate_at_5"] = (
            f"PASS: {hit_rate:.2f} >= target {slo.min_hit_rate_at_5:.2f}"
        )
    else:
        report["hit_rate_at_5"] = (
            f"FAIL: {hit_rate:.2f} < target {slo.min_hit_rate_at_5:.2f}"
        )
        passed = False

    # 2. Faithfulness
    faithfulness = metrics.get("faithfulness_score", 0.0)
    if faithfulness >= slo.min_faithfulness_score:
        report["faithfulness_score"] = (
            f"PASS: {faithfulness:.2f} >= target {slo.min_faithfulness_score:.2f}"
        )
    else:
        report["faithfulness_score"] = (
            f"FAIL: {faithfulness:.2f} < target {slo.min_faithfulness_score:.2f}"
        )
        passed = False

    # 3. p95 Latency
    p95_lat = metrics.get("p95_latency_seconds", float("inf"))
    if p95_lat <= slo.max_p95_latency_seconds:
        report["p95_latency_seconds"] = (
            f"PASS: {p95_lat:.2f}s <= target {slo.max_p95_latency_seconds:.2f}s"
        )
    else:
        report["p95_latency_seconds"] = (
            f"FAIL: {p95_lat:.2f}s > target {slo.max_p95_latency_seconds:.2f}s"
        )
        passed = False

    # 4. Hallucination Rate
    hallucination_rate = metrics.get("hallucination_rate", 1.0)
    if hallucination_rate <= slo.max_hallucination_rate:
        report["hallucination_rate"] = (
            f"PASS: {hallucination_rate:.2%} <= target {slo.max_hallucination_rate:.2%}"
        )
    else:
        report["hallucination_rate"] = (
            f"FAIL: {hallucination_rate:.2%} > target {slo.max_hallucination_rate:.2%}"
        )
        passed = False

    # 5. Cost per Query
    cost = metrics.get("cost_per_query_usd", float("inf"))
    if cost <= slo.max_cost_per_query_usd:
        report["cost_per_query_usd"] = (
            f"PASS: ${cost:.5f} <= target ${slo.max_cost_per_query_usd:.5f}"
        )
    else:
        report["cost_per_query_usd"] = (
            f"FAIL: ${cost:.5f} > target ${slo.max_cost_per_query_usd:.5f}"
        )
        passed = False

    return passed, report
