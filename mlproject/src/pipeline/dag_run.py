"""CLI entry point for running ML pipelines.

This module provides a command-line interface for executing different
pipeline types: train, eval, test, and tune.

Usage
-----
Training::

    python -m mlproject.src.pipeline.dag_run train \
        --config mlproject/configs/experiments/train_xgboost.yaml

Evaluation::

    python -m mlproject.src.pipeline.dag_run eval \
        --config mlproject/configs/experiments/eval_xgboost.yaml

Testing/Inference::

    python -m mlproject.src.pipeline.dag_run test \
        --config mlproject/configs/experiments/test_xgboost.yaml

Hyperparameter Tuning::

    python -m mlproject.src.pipeline.dag_run tune \
        --config configs/experiments/tune_xgboost.yaml \
        --trials 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from mlproject.src.pipeline.flexible_training import FlexibleTrainingPipeline

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"


def run_training(cfg_path: str) -> None:
    """Run training pipeline.

    Parameters
    ----------
    cfg_path : str
        Path to experiment configuration YAML.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"\n{'='*60}")
    print("[RUN] Starting TRAINING pipeline")
    print(f"[RUN] Config: {cfg_path}")
    print(f"{'='*60}\n")

    pipeline = FlexibleTrainingPipeline(cfg_path)
    context = pipeline.run_exp()

    print(f"\n{'='*60}")
    print("[RUN] Training COMPLETE")
    if "evaluate_metrics" in context:
        print("[RUN] Final metrics:")
        for metric, value in context["evaluate_metrics"].items():
            print(f"  - {metric}: {value:.4f}")
    print(f"{'='*60}\n")


def run_eval(cfg_path: str, model_path: Optional[str] = None) -> None:
    """Run evaluation pipeline.

    Parameters
    ----------
    cfg_path : str
        Path to experiment configuration YAML.
    model_path : str, optional
        Path to trained model file (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"\n{'='*60}")
    print("[RUN] Starting EVALUATION pipeline")
    print(f"[RUN] Config: {cfg_path}")
    if model_path:
        print(f"[RUN] Model override: {model_path}")
    print(f"{'='*60}\n")

    pipeline = FlexibleTrainingPipeline(cfg_path)
    context = pipeline.run_exp()

    print(f"\n{'='*60}")
    print("[RUN] Evaluation COMPLETE")
    if "evaluate_metrics" in context:
        print("[RUN] Evaluation metrics:")
        for metric, value in context["evaluate_metrics"].items():
            print(f"  - {metric}: {value:.4f}")
    print(f"{'='*60}\n")


def run_test(
    cfg_path: str,
) -> None:
    """Run test/inference pipeline.

    Parameters
    ----------
    cfg_path : str
        Path to experiment configuration YAML.
    output_path : str, optional
        Path to save predictions (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"\n{'='*60}")
    print("[RUN] Starting TEST/INFERENCE pipeline")
    print(f"[RUN] Config: {cfg_path}")

    pipeline = FlexibleTrainingPipeline(cfg_path)
    context = pipeline.run_exp()

    print(f"\n{'='*60}")
    print("[RUN] Test/Inference COMPLETE")

    # Check for predictions in context
    pred_keys = [k for k in context.keys() if "_predictions" in k]
    if pred_keys:
        for key in pred_keys:
            preds = context[key]
            print(f"[RUN] {key}: {len(preds)} predictions generated")


def run_tune(cfg_path: str, n_trials: Optional[int] = None) -> None:
    """Run hyperparameter tuning pipeline.

    Parameters
    ----------
    cfg_path : str
        Path to experiment configuration YAML.
    n_trials : int, optional
        Number of trials to run (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print(f"\n{'='*60}")
    print("[RUN] Starting HYPERPARAMETER TUNING pipeline")
    print(f"[RUN] Config: {cfg_path}")
    if n_trials:
        print(f"[RUN] Trials override: {n_trials}")
    print(f"{'='*60}\n")

    pipeline = FlexibleTrainingPipeline(cfg_path)
    context = pipeline.run_exp()

    print(f"\n{'='*60}")
    print("[RUN] Tuning COMPLETE")

    # Check for best params in context
    param_keys = [k for k in context.keys() if "_best_params" in k]
    if param_keys:
        for key in param_keys:
            params = context[key]
            print(f"[RUN] Best parameters ({key}):")
            for param, value in params.items():
                print(f"  - {param}: {value}")

    # Check for best metric
    study_keys = [k for k in context.keys() if "_study" in k]
    if study_keys:
        study = context[study_keys[0]]
        print(f"[RUN] Best metric value: {study.best_value:.4f}")

    print(f"{'='*60}\n")


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run ML pipelines (train/eval/test/tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  %(prog)s train --config configs/experiments/xgboost.yaml

  # Evaluation
  %(prog)s eval --config configs/experiments/eval_xgb.yaml

  # Inference
  %(prog)s test --config configs/experiments/test.yaml \\
      --output outputs/predictions.csv

  # Tuning
  %(prog)s tune --config configs/experiments/tune_xgb.yaml \\
      --trials 100
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment config YAML",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment config YAML",
    )
    eval_parser.add_argument(
        "--model",
        "-m",
        help="Path to model file (overrides config)",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test/inference pipeline")
    test_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment config YAML",
    )
    test_parser.add_argument(
        "--output",
        "-o",
        help="Path to save predictions (overrides config)",
    )

    # Tune command
    tune_parser = subparsers.add_parser(
        "tune", help="Run hyperparameter tuning pipeline"
    )
    tune_parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to experiment config YAML",
    )
    tune_parser.add_argument(
        "--trials",
        "-t",
        type=int,
        help="Number of tuning trials (overrides config)",
    )

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == "train":
            run_training(args.config)
        elif args.command == "eval":
            run_eval(args.config, args.model)
        elif args.command == "test":
            run_test(args.config)
        elif args.command == "tune":
            run_tune(args.config, args.trials)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
