#!/usr/bin/env python
"""CLI executable entrypoint to run the AniRec evaluation suite."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.runner import run_evaluation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AniRec Evaluation Benchmark Suite")
    parser.add_argument(
        "--experiment",
        type=str,
        default="production_pipeline",
        help="Experiment identifier name (e.g. baseline, tuned_reranker)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of queries to test",
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
        help="Output directory for evaluation run JSON reports",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation_suite(
            experiment_name=args.experiment,
            max_samples=args.max_samples,
            golden_path=args.golden_set,
            out_dir=args.out_dir,
            verbose=True,
        )
    )


if __name__ == "__main__":
    main()
