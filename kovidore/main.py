#!/usr/bin/env python3
"""
KoVidore Benchmark - Korean Vision Document Retrieval Evaluation

This script demonstrates how to evaluate models on the KoVidore benchmark.
"""

import argparse
import logging
from .evaluate import run_benchmark, ALL_TASKS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run KoVidore benchmark evaluation")
    parser.add_argument(
        "--model", 
        type=str, 
        default="average_word_embeddings_komninos",
        help="Model name to evaluate (default: average_word_embeddings_komninos)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help=f"Tasks to run (default: all). Available: {', '.join(ALL_TASKS)}"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding (default: 16)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results already exist"
    )
    
    args = parser.parse_args()
    
    if args.list_tasks:
        logger.info("Available tasks:")
        for task in ALL_TASKS:
            logger.info(f"  - {task}")
        return
    
    tasks_to_run = args.tasks if args.tasks else ALL_TASKS
    
    logger.info("Starting KoVidore benchmark evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Tasks: {', '.join(tasks_to_run)}")
    logger.info("=" * 60)
    
    evaluation = run_benchmark(
        args.model,
        tasks_to_run,
        batch_size=args.batch_size,
        skip_existing=not args.force
    )
    
    if evaluation is not None:
        logger.info("Benchmark completed successfully!")
        logger.info("Results have been saved to the results directory.")
    else:
        logger.error("Benchmark failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
