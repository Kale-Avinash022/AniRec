"""Eval Benchmark Comparison Reporter for AniRec (Step 53).

Parses historic JSON evaluation run reports from `evals/results/` and renders
structured Markdown comparison tables and diff summaries for documentation,
dashboards, and GitHub READMEs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_all_eval_reports(
    results_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Load and parse all valid evaluation JSON reports in the target results directory.

    Args:
        results_dir: Path to directory containing evaluation JSON files.

    Returns:
        List of parsed report dictionaries sorted chronologically by file creation/timestamp.
    """
    target_dir = Path(results_dir) if results_dir else Path("evals/results")
    if not target_dir.exists():
        return []

    json_files = sorted(target_dir.glob("*.json"))
    reports: List[Dict[str, Any]] = []

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, dict) and "summary" in data:
                    # Inject filepath into metadata
                    data["_filepath"] = str(f)
                    reports.append(data)
        except Exception:
            continue

    return reports


def generate_comparison_table(
    results_dir: Optional[str | Path] = None,
    reports: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a GitHub-flavored Markdown comparison table across benchmark runs.

    Args:
        results_dir: Directory containing evaluation report JSON files.
        reports: Optional pre-loaded list of report dictionaries.

    Returns:
        Markdown table string.
    """
    if reports is None:
        reports = load_all_eval_reports(results_dir)

    if not reports:
        return "No evaluation run reports found in results directory."

    table_lines: List[str] = [
        "| Pipeline Version / Experiment | Queries | Hit Rate @ 5 | MRR | Faithfulness | Constraint Pass | p50 Latency | p95 Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in reports:
        s = r.get("summary", {})
        name = s.get("experiment_name", "Unknown")
        total_q = s.get("total_queries", 0)
        hit_5 = s.get("hit_rate_at_5", 0.0)
        mrr = s.get("mrr", 0.0)
        faith_rate = s.get("faithfulness_rate", 0.0)
        faith_score = s.get("avg_faithfulness_score", 0.0)
        const_pass = s.get("constraint_satisfaction_rate", 0.0)
        p50_ms = s.get("p50_latency_ms", 0.0)
        p95_ms = s.get("p95_latency_ms", 0.0)

        faith_display = f"{faith_rate * 100:.1f}% ({faith_score}/5)" if faith_score else f"{faith_rate * 100:.1f}%"
        const_display = f"{const_pass * 100:.1f}%" if const_pass else "N/A"

        table_lines.append(
            f"| **{name}** | `{total_q}` | `{hit_5 * 100:.1f}%` | `{mrr:.3f}` | `{faith_display}` | `{const_display}` | `{p50_ms:.0f}ms` | `{p95_ms:.0f}ms` |"
        )

    return "\n".join(table_lines)


def generate_comparison_summary(
    results_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generate structured summary dictionary across all evaluated runs.

    Args:
        results_dir: Target evaluation results folder.

    Returns:
        Dict with total_runs, best_hit_rate, best_mrr, best_faithfulness, and runs list.
    """
    reports = load_all_eval_reports(results_dir)
    if not reports:
        return {
            "total_runs": 0,
            "best_hit_rate_at_5": 0.0,
            "best_mrr": 0.0,
            "best_faithfulness_rate": 0.0,
            "runs": [],
        }

    summaries = [r.get("summary", {}) for r in reports]
    best_hit = max((s.get("hit_rate_at_5", 0.0) for s in summaries), default=0.0)
    best_mrr = max((s.get("mrr", 0.0) for s in summaries), default=0.0)
    best_faith = max((s.get("faithfulness_rate", 0.0) for s in summaries), default=0.0)

    return {
        "total_runs": len(reports),
        "best_hit_rate_at_5": best_hit,
        "best_mrr": best_mrr,
        "best_faithfulness_rate": best_faith,
        "runs": summaries,
    }


def save_comparison_markdown(
    output_path: str | Path,
    results_dir: Optional[str | Path] = None,
) -> Path:
    """Save the Markdown comparison table to a markdown document file.

    Args:
        output_path: Target markdown destination path.
        results_dir: Target evaluation results folder.

    Returns:
        Path to written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_md = generate_comparison_table(results_dir)

    content = f"# AniRec Evaluation Benchmark Comparison Report\n\n{table_md}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
