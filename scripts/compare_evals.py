#!/usr/bin/env python
"""CLI executable script to render benchmark comparisons from historic evaluation runs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.reporter import generate_comparison_table, save_comparison_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AniRec Evaluation Comparison Table")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="evals/results",
        help="Directory containing evaluation run JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output markdown file path (e.g. evals/COMPARISON.md)",
    )
    args = parser.parse_args()

    table_md = generate_comparison_table(results_dir=args.results_dir)
    print("\n" + table_md + "\n")

    if args.output:
        saved_path = save_comparison_markdown(output_path=args.output, results_dir=args.results_dir)
        print(f"✅ Saved comparison report to {saved_path}")


if __name__ == "__main__":
    main()
