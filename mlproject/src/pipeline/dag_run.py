"""CLI entry point for running ML pipelines.

This module provides a command-line interface for executing different
pipeline types: train, eval, serve, and tune.

Key Feature: Separation of pipeline structure and experiment config.
- Pipeline configs define step DAG (standard_train.yaml, standard_tune.yaml)
- Experiment configs define data/model/hyperparams (xgboost.yaml, lstm.yaml)

Usage
-----
Training::

    python -m mlproject.src.pipeline.dag_run train \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_train.yaml

    python -m mlproject.src.pipeline.dag_run train \
        -e mlproject/configs/experiments/etth3.yaml \
        -p mlproject/configs/pipelines/standard_train.yaml

Evaluation::

    python -m mlproject.src.pipeline.dag_run eval \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_eval.yaml \
        --alias latest

    python -m mlproject.src.pipeline.dag_run eval \
        --e mlproject/configs/experiments/etth3.yaml \
        --p mlproject/configs/pipelines/standard_eval.yaml \
        --a latest

Serving::

    python -m mlproject.src.pipeline.dag_run serve \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_serve.yaml \
        --input ./sample_input.csv \
        --alias latest

    python -m mlproject.src.pipeline.dag_run serve \
        --experiment mlproject/configs/experiments/etth3_feast.yaml \
        --pipeline mlproject/configs/pipelines/standard_serve.yaml \
        --alias latest \
        --time_point "now"



Hyperparameter Tuning::

    python -m mlproject.src.pipeline.dag_run tune \
        --experiment mlproject/configs/experiments/etth3.yaml \
        --pipeline mlproject/configs/pipelines/standard_tune.yaml \
        --trials 50
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.pipeline.flexible_pipeline import FlexiblePipeline
from mlproject.src.utils.config_loader import ConfigLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

# Default pipeline paths
DEFAULT_PIPELINES = {
    "train": "mlproject/configs/pipelines/standard_train.yaml",
    "eval": "mlproject/configs/pipelines/standard_eval.yaml",
    "serve": "mlproject/configs/pipelines/standard_serve.yaml",
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
        Pipeline mode (train/eval/serve/tune).

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
    # Load experiment config with defaults resolved
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

    # Merge: experiment < pipeline (pipeline has priority)
    merged = OmegaConf.merge(exp_cfg, pipe_cfg)
    if isinstance(merged, ListConfig):
        merged = OmegaConf.create({"config": merged})
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


def _load_csv_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Parameters
    ----------
    input_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[SERVE] Loading data from CSV: {input_path}")
    df = pd.read_csv(input_path)

    # Handle date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    print(f"[SERVE] Loaded CSV data shape: {df.shape}")
    return df


def _load_feast_data(cfg: DictConfig, time_point: str) -> pd.DataFrame:
    """Load data from Feast Feature Store.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with Feast settings.
    time_point : str
        Timestamp for Feast (ISO format or "now").

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    ValueError
        If Feast URI not configured.
    """
    uri = cfg.data.get("path", "")
    if not uri.startswith("feast://"):
        raise ValueError(
            "Serving without input file requires Feast URI in config. "
            "Set data.path to 'feast://...' or provide --input argument."
        )

    print(f"[SERVE] Loading data from Feast: {uri}")
    print(f"[SERVE] Time point: {time_point}")

    facade = FeatureStoreFacade(cfg, mode="online")
    df = facade.load_features(time_point=time_point)

    print(f"[SERVE] Loaded Feast data shape: {df.shape}")
    return df


def prepare_serving_data(
    cfg: DictConfig, input_path: Optional[str] = None, time_point: str = "now"
) -> Dict[str, Any]:
    """Prepare data for serving mode from CSV or Feast.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with Feast settings.
    input_path : str, optional
        Path to input CSV file. If None, loads from Feast.
    time_point : str, default="now"
        Timestamp for Feast (ISO format or "now").

    Returns
    -------
    Dict[str, Any]
        Context dict with keys expected by PreprocessingStep:
        - df: Full dataframe
        - train_df: Empty (not used in serving)
        - val_df: Empty (not used in serving)
        - test_df: Same as df (will be preprocessed)
        - is_splited_input: False

    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist.
    ValueError
        If neither CSV nor Feast URI is provided.
    """
    # Load data from CSV or Feast
    if input_path:
        df = _load_csv_data(input_path)
    else:
        df = _load_feast_data(cfg, time_point)

    # Mimic DataLoaderStep output structure
    context = {
        "df": df.copy(),
        "train_df": pd.DataFrame(),  # Empty (not used in serving)
        "val_df": pd.DataFrame(),  # Empty (not used in serving)
        "test_df": df.copy(),  # Will be preprocessed
        "is_splited_input": False,
    }

    return context


def run_training(experiment_path: str, pipeline_path: Optional[str] = None) -> None:
    """Run training pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting TRAINING pipeline")
    print(f"{'='*60}\n")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="train")
    temp_config = ".temp_merged_train.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        print(f"\n{'='*60}")
        print("[RUN] Training COMPLETE")
        if "evaluate_metrics" in context:
            print("[RUN] Final metrics:")
            for metric, value in context["evaluate_metrics"].items():
                print(f"  - {metric}: {value:.4f}")
        print(f"{'='*60}\n")

    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_eval(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    alias: str = "latest",
) -> None:
    """Run evaluation pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    alias : str, default="latest"
        Model alias in MLflow Registry.
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting EVALUATION pipeline")
    print(f"[RUN] Model alias: {alias}")
    print(f"{'='*60}\n")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="eval")

    # Update alias for model_loader AND preprocessor
    for step in merged_cfg.pipeline.steps:
        step_type = step.get("type", "")
        # Update model_loader
        if step_type == "model_loader":
            step.alias = alias
            print(f"[CONFIG] Override: model_loader alias='{alias}'")
        # Update preprocessor (only if is_train=False)
        elif step_type == "preprocessor" and not step.get("is_train", True):
            step.alias = alias
            print(f"[CONFIG] Override: preprocessor alias='{alias}'")

    temp_config = ".temp_merged_eval.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        print(f"\n{'='*60}")
        print("[RUN] Evaluation COMPLETE")
        if "evaluate_metrics" in context:
            print("[RUN] Evaluation metrics:")
            for metric, value in context["evaluate_metrics"].items():
                print(f"  - {metric}: {value:.4f}")
        print(f"{'='*60}\n")

    finally:
        Path(temp_config).unlink(missing_ok=True)


def _check_initial_context_support(pipeline: FlexiblePipeline) -> bool:
    """Check if pipeline supports initial_context parameter.

    Parameters
    ----------
    pipeline : FlexibleTrainingPipeline
        Pipeline instance to check.

    Returns
    -------
    bool
        True if supports initial_context.

    Raises
    ------
    AttributeError
        If pipeline missing run_exp method.
    """
    if not hasattr(pipeline, "run_exp"):
        raise AttributeError("Pipeline missing run_exp() method")

    sig = inspect.signature(pipeline.run_exp)
    return "initial_context" in sig.parameters


def _run_pipeline_with_context(
    pipeline: FlexiblePipeline, initial_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Run pipeline with initial context (with fallback).

    Parameters
    ----------
    pipeline : FlexibleTrainingPipeline
        Pipeline instance.
    initial_context : Dict[str, Any]
        Pre-initialized context.

    Returns
    -------
    Dict[str, Any]
        Pipeline execution context.

    Raises
    ------
    NotImplementedError
        If pipeline doesn't support context injection.
    """
    if _check_initial_context_support(pipeline):
        # Direct support
        return pipeline.run_exp(initial_context=initial_context)

    # Fallback: Set context before run
    if hasattr(pipeline, "executor") and hasattr(pipeline.executor, "context"):
        pipeline.executor.context = initial_context
        return pipeline.run_exp()

    raise NotImplementedError(
        "FlexibleTrainingPipeline.run_exp() does not support "
        "initial_context parameter. "
        "Please update flexible_training.py:\n"
        "def run_exp(self, initial_context=None):\n"
        "    context = initial_context or {}\n"
        "    ..."
    )


def run_serve(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    input_path: Optional[str] = None,
    alias: str = "latest",
    time_point: str = "now",
) -> Any:
    """Run serve pipeline with runtime data injection.

    This function supports two serving modes:
    1. CSV Mode: Load data from file
    2. Feast Mode: Load data from Feature Store

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    input_path : str, optional
        Path to input CSV file. If None, loads from Feast.
    alias : str, default="latest"
        Model alias in MLflow Registry.
    time_point : str, default="now"
        Timestamp for Feast (ISO format or "now").

    Returns
    -------
    np.ndarray
        Model predictions.

    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist.
    ValueError
        If neither CSV nor Feast URI provided.

    Examples
    --------
    # CSV Mode
    python -m mlproject.src.pipeline.dag_run serve \
        --experiment config.yaml \
        --input data.csv \
        --alias production

    # Feast Mode
    python -m mlproject.src.pipeline.dag_run serve \
        --experiment config.yaml \
        --alias production \
        --time_point "2024-01-01T12:00:00"
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting SERVING pipeline")
    print(f"[RUN] Model alias: {alias}")
    if input_path:
        print(f"[RUN] Data source: CSV ({input_path})")
    else:
        print("[RUN] Data source: Feast Feature Store")
        print(f"[RUN] Time point: {time_point}")
    print(f"{'='*60}\n")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="serve")

    # Update model alias
    if alias != "latest":
        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "model_loader":
                step.alias = alias

    temp_config = ".temp_merged_serve.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)

        # Pre-initialize context with data
        initial_context = prepare_serving_data(
            cfg=merged_cfg, input_path=input_path, time_point=time_point
        )

        print(f"[SERVE] Pre-initialized context with {len(initial_context)} keys")

        # Run pipeline with initial context
        context = _run_pipeline_with_context(pipeline, initial_context)

        print(f"\n{'='*60}")
        print("[RUN] Serving COMPLETE")

        # Get predictions
        pred_keys = [k for k in context.keys() if "_predictions" in k]
        if not pred_keys:
            raise RuntimeError("No predictions found in pipeline context")

        predictions = context[pred_keys[0]]
        print(f"[RUN] Generated {len(predictions)} predictions")

        if hasattr(predictions, "shape"):
            print(f"[RUN] Prediction shape: {predictions.shape}")

        if hasattr(predictions, "flatten") and len(predictions) > 0:
            preview_len = min(10, len(predictions))
            print(f"[RUN] First {preview_len}: {predictions[:preview_len]}")

        print(f"{'='*60}\n")
        return predictions

    finally:
        Path(temp_config).unlink(missing_ok=True)


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
    """
    print(f"\n{'='*60}")
    print("[RUN] Starting HYPERPARAMETER TUNING pipeline")
    if n_trials:
        print(f"[RUN] Trials override: {n_trials}")
    print(f"{'='*60}\n")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="tune")

    # Override n_trials if provided
    if n_trials:
        if "tuning" not in merged_cfg:
            merged_cfg.tuning = {}
        merged_cfg.tuning.n_trials = n_trials

        # Update step config
        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "tuning":
                step.n_trials = n_trials
                print(f"[CONFIG] Override: n_trials={n_trials} in step '{step.id}'")
                break

    temp_config = ".temp_merged_tune.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        print(f"\n{'='*60}")
        print("[RUN] Tuning COMPLETE")

        param_keys = [k for k in context.keys() if "_best_params" in k]
        if param_keys:
            for key in param_keys:
                params = context[key]
                print(f"[RUN] Best parameters ({key}):")
                for param, value in params.items():
                    print(f"  - {param}: {value}")

        study_keys = [k for k in context.keys() if "_study" in k]
        if study_keys:
            study = context[study_keys[0]]
            print(f"[RUN] Best metric value: {study.best_value:.4f}")

        print(f"{'='*60}\n")

    finally:
        Path(temp_config).unlink(missing_ok=True)


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run ML pipelines (train/eval/serve/tune)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  %(prog)s train --experiment configs/experiments/xgboost.yaml

  # Evaluation
  %(prog)s eval --experiment configs/experiments/xgboost.yaml --alias production

  # Serving with CSV input (runtime data injection)
  %(prog)s serve --experiment configs/experiments/xgboost.yaml \
      --input data/test.csv --alias production

  # Tuning
  %(prog)s tune --experiment configs/experiments/xgboost.yaml --trials 100
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    train_parser.add_argument(
        "--pipeline", "-p", help="Pipeline config YAML (default: standard_train.yaml)"
    )

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    eval_parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    eval_parser.add_argument(
        "--pipeline", "-p", help="Pipeline config YAML (default: standard_eval.yaml)"
    )
    eval_parser.add_argument(
        "--alias",
        "-a",
        default="latest",
        help="Model alias (latest/production/staging)",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run serve pipeline")
    serve_parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    serve_parser.add_argument(
        "--pipeline", "-p", help="Pipeline config YAML (default: standard_serve.yaml)"
    )
    serve_parser.add_argument(
        "--input", "-i", help="Input CSV file (if omitted, uses Feast)"
    )
    serve_parser.add_argument("--alias", "-a", default="latest", help="Model alias")
    serve_parser.add_argument(
        "--time_point",
        "-t",
        default="now",
        help="Timestamp for Feast (ISO format or 'now')",
    )

    # Tune command
    tune_parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    tune_parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    tune_parser.add_argument(
        "--pipeline", "-p", help="Pipeline config YAML (default: standard_tune.yaml)"
    )
    tune_parser.add_argument("--trials", "-t", type=int, help="Number of tuning trials")

    args = parser.parse_args()

    # Execute command
    try:
        if args.command == "train":
            run_training(args.experiment, args.pipeline)
        elif args.command == "eval":
            run_eval(args.experiment, args.pipeline, args.alias)
        elif args.command == "serve":
            run_serve(
                args.experiment, args.pipeline, args.input, args.alias, args.time_point
            )
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
