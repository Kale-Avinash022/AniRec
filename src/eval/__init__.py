"""Evaluation suite package for AniRec."""

from src.eval.dataset import (
    GoldenDataset,
    GoldenQueryItem,
    load_golden_set,
    save_golden_set,
)
from src.eval.retrieval_metrics import (
    RetrievalMetricsSummary,
    calculate_hit_at_k,
    calculate_mrr,
    calculate_precision_at_k,
    calculate_recall_at_k,
    evaluate_retrieval_batch,
)
from src.eval.judge import (
    ConstraintSatisfactionVerdict,
    FaithfulnessVerdict,
    GenerationEvalResult,
    evaluate_generation_batch,
    judge_constraint_satisfaction,
    judge_faithfulness,
)
from src.eval.runner import run_evaluation_suite
from src.eval.reporter import (
    generate_comparison_summary,
    generate_comparison_table,
    load_all_eval_reports,
    save_comparison_markdown,
)
from src.eval.experiments import (
    AblationMode,
    run_ablation_experiment,
    run_all_ablations,
)

__all__ = [
    "GoldenDataset",
    "GoldenQueryItem",
    "load_golden_set",
    "save_golden_set",
    "RetrievalMetricsSummary",
    "calculate_hit_at_k",
    "calculate_mrr",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "evaluate_retrieval_batch",
    "ConstraintSatisfactionVerdict",
    "FaithfulnessVerdict",
    "GenerationEvalResult",
    "evaluate_generation_batch",
    "judge_constraint_satisfaction",
    "judge_faithfulness",
    "run_evaluation_suite",
    "generate_comparison_summary",
    "generate_comparison_table",
    "load_all_eval_reports",
    "save_comparison_markdown",
    "AblationMode",
    "run_ablation_experiment",
    "run_all_ablations",
]
