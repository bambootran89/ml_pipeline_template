"""CLI entry point for running ML pipelines.

This module provides a command-line interface for executing different
pipeline types: train, eval, test, and tune.

Key Feature: Separation of pipeline structure and experiment config.
- Pipeline configs define step DAG (standard_train.yaml, standard_tune.yaml)
- Experiment configs define data/model/hyperparams (xgboost.yaml, lstm.yaml)

Usage
-----
Training::

    python -m mlproject.src.pipeline.dag_run train \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_train.yaml

Evaluation::

    python -m mlproject.src.pipeline.dag_run eval \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_eval.yaml

Testing/Inference::

    python -m mlproject.src.pipeline.dag_run test \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_test.yaml

Hyperparameter Tuning::

    python -m mlproject.src.pipeline.dag_run tune \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_tune.yaml \
        --trials 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.flexible_training import FlexibleTrainingPipeline
from mlproject.src.utils.config_loader import ConfigLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

# Default pipeline paths
DEFAULT_PIPELINES = {
    "train": "mlproject/configs/pipelines/standard_train.yaml",
    "eval": "mlproject/configs/pipelines/standard_eval.yaml",
    "test": "mlproject/configs/pipelines/standard_test.yaml",
    "tune": "mlproject/configs/pipelines/standard_tune.yaml",
}


def merge_configs(
    experiment_path: str, pipeline_path: Optional[str] = None, mode: str = "train"
) -> DictConfig:
    """Merge experiment config with pipeline config.

    Uses ConfigLoader to resolve defaults in experiment config,
    then merges with pipeline config.

    Parameters
    ----------
    experiment_path : str
        Path to experiment config (data, model, hyperparams).
    pipeline_path : str, optional
        Path to pipeline config (step DAG).
        If None, uses default for mode.
    mode : str
        Pipeline mode (train/eval/test/tune).

    Returns
    -------
    DictConfig
        Merged configuration ready for pipeline execution.

    Raises
    ------
    FileNotFoundError
        If config files don't exist.
    ValueError
        If no default pipeline exists for mode.
    """
    # Load experiment config with defaults resolved (using ConfigLoader)
    exp_cfg = ConfigLoader.load(experiment_path)

    # Load pipeline config
    if pipeline_path is None:
        pipeline_path = DEFAULT_PIPELINES.get(mode)
        if pipeline_path is None:
            raise ValueError(f"No default pipeline for mode: {mode}")

    pipe_file = Path(pipeline_path)
    if not pipe_file.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")

    pipe_cfg = OmegaConf.load(pipe_file)

    # Merge: experiment config is base, pipeline config overrides
    merged = OmegaConf.merge(exp_cfg, pipe_cfg)

    print("\n[CONFIG] Merged configuration:")
    print(f"  - Experiment: {experiment_path}")
    print(f"  - Pipeline:   {pipeline_path}")

    return merged


def save_merged_config(cfg: DictConfig, output_path: str) -> None:
    """Save merged config to temporary file.

    Parameters
    ----------
    cfg : DictConfig
        Merged configuration.
    output_path : str
        Path to save config.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    print(f"  - Merged saved: {output_path}\n")


def run_training(experiment_path: str, pipeline_path: Optional[str] = None) -> None:
    """Run training pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting TRAINING pipeline")
    print(f"{'='*60}\n")

    # Merge configs
    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="train")

    # Save to temp file for FlexibleTrainingPipeline
    temp_config = ".temp_merged_train.yaml"
    save_merged_config(merged_cfg, temp_config)

    # Run pipeline
    pipeline = FlexibleTrainingPipeline(temp_config)
    context = pipeline.run_exp()

    # Cleanup temp file
    Path(temp_config).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("[RUN] Training COMPLETE")
    if "evaluate_metrics" in context:
        print("[RUN] Final metrics:")
        for metric, value in context["evaluate_metrics"].items():
            print(f"  - {metric}: {value:.4f}")
    print(f"{'='*60}\n")


def run_eval(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> None:
    """Run evaluation pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    model_path : str, optional
        Path to trained model file (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting EVALUATION pipeline")
    if model_path:
        print(f"[RUN] Model override: {model_path}")
    print(f"{'='*60}\n")

    # Merge configs
    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="eval")

    # Override model path if provided
    if model_path:
        # Find model_loader step and update
        for step in merged_cfg.pipeline.steps:
            if step.type == "model_loader":
                step.model_path = model_path

    # Save to temp file
    temp_config = ".temp_merged_eval.yaml"
    save_merged_config(merged_cfg, temp_config)

    # Run pipeline
    pipeline = FlexibleTrainingPipeline(temp_config)
    context = pipeline.run_exp()

    # Cleanup
    Path(temp_config).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("[RUN] Evaluation COMPLETE")
    if "evaluate_metrics" in context:
        print("[RUN] Evaluation metrics:")
        for metric, value in context["evaluate_metrics"].items():
            print(f"  - {metric}: {value:.4f}")
    print(f"{'='*60}\n")


def run_test(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """Run test/inference pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    output_path : str, optional
        Path to save predictions (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting TEST/INFERENCE pipeline")
    if output_path:
        print(f"[RUN] Output override: {output_path}")
    print(f"{'='*60}\n")

    # Merge configs
    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="test")

    # Override output path if provided
    if output_path:
        for step in merged_cfg.pipeline.steps:
            if step.type == "inference":
                step.save_path = output_path

    # Save to temp file
    temp_config = ".temp_merged_test.yaml"
    save_merged_config(merged_cfg, temp_config)

    # Run pipeline
    pipeline = FlexibleTrainingPipeline(temp_config)
    context = pipeline.run_exp()

    # Cleanup
    Path(temp_config).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("[RUN] Test/Inference COMPLETE")

    # Check for predictions in context
    pred_keys = [k for k in context.keys() if "_predictions" in k]
    if pred_keys:
        for key in pred_keys:
            preds = context[key]
            print(f"[RUN] {key}: {len(preds)} predictions generated")
    print(f"{'='*60}\n")


def run_tune(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    n_trials: Optional[int] = None,
) -> None:
    """Run hyperparameter tuning pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    n_trials : int, optional
        Number of trials to run (overrides config).

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting HYPERPARAMETER TUNING pipeline")
    if n_trials:
        print(f"[RUN] Trials override: {n_trials}")
    print(f"{'='*60}\n")

    # Merge configs
    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="tune")

    # Override n_trials if provided
    if n_trials:
        # Method 1: Update tuning config (read by TuningStep.execute)
        if "tuning" not in merged_cfg:
            merged_cfg.tuning = {}
        merged_cfg.tuning.n_trials = n_trials

        # Method 2: Update step config (read by TuningStep.__init__)
        # Find tune step and update n_trials parameter
        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "tuning":
                step.n_trials = n_trials
                print(f"[CONFIG] Override: n_trials={n_trials} in step '{step.id}'")
                break

    # Save to temp file
    temp_config = ".temp_merged_tune.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        # Run pipeline
        pipeline = FlexibleTrainingPipeline(temp_config)
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

    finally:
        # Cleanup
        Path(temp_config).unlink(missing_ok=True)


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run ML pipelines (train/eval/test/tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training with default pipeline
  %(prog)s train --experiment configs/experiments/xgboost.yaml

  # Training with custom pipeline
  %(prog)s train \
      --experiment configs/experiments/xgboost.yaml \
      --pipeline configs/pipelines/custom_train.yaml

  # Evaluation
  %(prog)s eval --experiment configs/experiments/xgboost.yaml

  # Inference
  %(prog)s test --experiment configs/experiments/xgboost.yaml

  # Tuning
  %(prog)s tune --experiment configs/experiments/xgboost.yaml --trials 100
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Path to experiment config YAML (data, model, hyperparams)",
    )
    train_parser.add_argument(
        "--pipeline",
        "-p",
        help="Path to pipeline config YAML (step DAG). Default: standard_train.yaml",
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Path to experiment config YAML",
    )
    eval_parser.add_argument(
        "--pipeline",
        "-p",
        help="Path to pipeline config YAML. Default: standard_eval.yaml",
    )
    eval_parser.add_argument(
        "--model",
        "-m",
        help="Path to model file (overrides config)",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Run test/inference pipeline")
    test_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Path to experiment config YAML",
    )
    test_parser.add_argument(
        "--pipeline",
        "-p",
        help="Path to pipeline config YAML. Default: standard_test.yaml",
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
        "--experiment",
        "-e",
        required=True,
        help="Path to experiment config YAML",
    )
    tune_parser.add_argument(
        "--pipeline",
        "-p",
        help="Path to pipeline config YAML. Default: standard_tune.yaml",
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
            run_training(args.experiment, args.pipeline)
        elif args.command == "eval":
            run_eval(args.experiment, args.pipeline, args.model)
        elif args.command == "test":
            run_test(args.experiment, args.pipeline, args.output)
        elif args.command == "tune":
            run_tune(args.experiment, args.pipeline, args.trials)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
