#!/usr/bin/env python
"""CLI executable script to execute all 4 ablation benchmarks sequentially."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.experiments import run_all_ablations
from src.eval.reporter import generate_comparison_table, save_comparison_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full Ablation Benchmark Progression")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on test queries per ablation mode",
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default=None,
        help="Custom path to golden evaluation dataset JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evals/results",
        help="Directory to save evaluation run JSON reports",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="evals/BENCHMARK_REPORT.md",
        help="Markdown comparison report output path",
    )
    args = parser.parse_args()

    asyncio.run(
        run_all_ablations(
            max_samples=args.max_samples,
            golden_path=args.golden_set,
            out_dir=args.out_dir,
            verbose=True,
        )
    )

    table_md = generate_comparison_table(results_dir=args.out_dir)
    print("\n" + "=" * 60)
    print("📈 ABLATION PROGRESSION COMPARISON TABLE")
    print("=" * 60)
    print(table_md)
    print("=" * 60 + "\n")

    if args.report_file:
        saved_path = save_comparison_markdown(output_path=args.report_file, results_dir=args.out_dir)
        print(f"✅ Saved comparison report to: {saved_path}")


if __name__ == "__main__":
    main()
