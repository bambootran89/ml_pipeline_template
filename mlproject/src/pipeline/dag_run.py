"""CLI entry point for running ML pipelines.

This module provides a command-line interface for executing different
pipeline types: train, eval, serve, tune, and generate-configs.
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
from mlproject.src.utils.config_generator import ConfigGenerator
from mlproject.src.utils.config_loader import ConfigLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

DEFAULT_PIPELINES: Dict[str, str] = {
    "train": "mlproject/configs/pipelines/standard_train.yaml",
    "eval": "mlproject/configs/pipelines/standard_eval.yaml",
    "serve": "mlproject/configs/pipelines/standard_serve.yaml",
    "tune": "mlproject/configs/pipelines/standard_tune.yaml",
}


def merge_configs(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    mode: str = "train",
) -> DictConfig:
    """Merge experiment config with pipeline config.

    Parameters
    ----------
    experiment_path : str
        Path to experiment config.
    pipeline_path : str, optional
        Path to pipeline config.
    mode : str
        Pipeline mode (train/eval/serve/tune).

    Returns
    -------
    DictConfig
        Merged configuration.

    Raises
    ------
    ValueError
        If no default pipeline for mode.
    FileNotFoundError
        If pipeline config not found.
    """
    exp_cfg = ConfigLoader.load(experiment_path)

    if pipeline_path is None:
        pipeline_path = DEFAULT_PIPELINES.get(mode)
        if pipeline_path is None:
            raise ValueError(f"No default pipeline for mode: {mode}")

    pipe_file = Path(pipeline_path)
    if not pipe_file.exists():
        raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")

    pipe_cfg = OmegaConf.load(pipe_file)
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
        If file not found.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[SERVE] Loading data from CSV: {input_path}")
    df = pd.read_csv(input_path)

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
        Timestamp for Feast.

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
        raise ValueError("Serving without input requires Feast URI in config.")

    print(f"[SERVE] Loading data from Feast: {uri}")
    print(f"[SERVE] Time point: {time_point}")

    facade = FeatureStoreFacade(cfg, mode="online")
    df = facade.load_features(time_point=time_point)

    print(f"[SERVE] Loaded Feast data shape: {df.shape}")
    return df


def prepare_serving_data(
    cfg: DictConfig,
    input_path: Optional[str] = None,
    time_point: str = "now",
) -> Dict[str, Any]:
    """Prepare data for serving mode.

    Parameters
    ----------
    cfg : DictConfig
        Configuration.
    input_path : str, optional
        Path to input CSV.
    time_point : str
        Timestamp for Feast.

    Returns
    -------
    Dict[str, Any]
        Context dict for pipeline.
    """
    df = _load_csv_data(input_path) if input_path else _load_feast_data(cfg, time_point)

    return {
        "df": df.copy(),
        "train_df": pd.DataFrame(),
        "val_df": pd.DataFrame(),
        "test_df": df.copy(),
        "is_splited_input": False,
    }


def _print_separator(title: str) -> None:
    """Print formatted separator with title."""
    print(f"\n{'=' * 60}")
    print(f"[RUN] {title}")
    print(f"{'=' * 60}\n")


def _print_metrics(context: Dict[str, Any], key: str = "evaluate_metrics") -> None:
    """Print metrics from context if available."""
    if key not in context:
        return
    print(f"[RUN] Metrics ({key}):")
    for metric, value in context[key].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.4f}")


def run_training(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
) -> None:
    """Run training pipeline.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    """
    _print_separator("Starting TRAINING pipeline")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="train")
    temp_config = ".temp_merged_train.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Training COMPLETE")
        _print_metrics(context)
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
    alias : str
        Model alias in MLflow Registry.
    """
    _print_separator("Starting EVALUATION pipeline")
    print(f"[RUN] Model alias: {alias}")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="eval")

    for step in merged_cfg.pipeline.steps:
        step_type = step.get("type", "")
        if step_type == "mlflow_loader":
            step.alias = alias
            print(f"[CONFIG] Override: mlflow_loader alias='{alias}'")
        elif step_type == "preprocessor" and not step.get("is_train", True):
            step.alias = alias
            print(f"[CONFIG] Override: preprocessor alias='{alias}'")

    temp_config = ".temp_merged_eval.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Evaluation COMPLETE")
        _print_metrics(context)
    finally:
        Path(temp_config).unlink(missing_ok=True)


def _check_initial_context_support(pipeline: FlexiblePipeline) -> bool:
    """Check if pipeline supports initial_context parameter.

    Parameters
    ----------
    pipeline : FlexiblePipeline
        Pipeline instance.

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
    pipeline: FlexiblePipeline,
    initial_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pipeline with initial context.

    Parameters
    ----------
    pipeline : FlexiblePipeline
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
        return pipeline.run_exp(initial_context=initial_context)

    if hasattr(pipeline, "executor") and hasattr(pipeline.executor, "context"):
        pipeline.executor.context = initial_context
        return pipeline.run_exp()

    raise NotImplementedError(
        "FlexiblePipeline.run_exp() does not support initial_context"
    )


def run_serve(
    experiment_path: str,
    pipeline_path: Optional[str] = None,
    input_path: Optional[str] = None,
    alias: str = "latest",
    time_point: str = "now",
) -> Any:
    """Run serve pipeline with runtime data injection.

    Parameters
    ----------
    experiment_path : str
        Path to experiment configuration YAML.
    pipeline_path : str, optional
        Path to pipeline structure YAML.
    input_path : str, optional
        Path to input CSV file.
    alias : str
        Model alias in MLflow Registry.
    time_point : str
        Timestamp for Feast.

    Returns
    -------
    Any
        Model predictions.

    Raises
    ------
    RuntimeError
        If no predictions found.
    """
    _print_separator("Starting SERVING pipeline")
    print(f"[RUN] Model alias: {alias}")

    if input_path:
        print(f"[RUN] Data source: CSV ({input_path})")
    else:
        print("[RUN] Data source: Feast Feature Store")
        print(f"[RUN] Time point: {time_point}")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="serve")

    if alias != "latest":
        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "mlflow_loader":
                step.alias = alias

    temp_config = ".temp_merged_serve.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        initial_context = prepare_serving_data(
            cfg=merged_cfg,
            input_path=input_path,
            time_point=time_point,
        )

        print(f"[SERVE] Pre-initialized context: {len(initial_context)} keys")
        context = _run_pipeline_with_context(pipeline, initial_context)

        _print_separator("Serving COMPLETE")

        pred_keys = [k for k in context if "_predictions" in k]
        if not pred_keys:
            raise RuntimeError("No predictions found in pipeline context")

        predictions = context[pred_keys[0]]
        print(f"[RUN] Generated {len(predictions)} predictions")

        if hasattr(predictions, "shape"):
            print(f"[RUN] Prediction shape: {predictions.shape}")

        if hasattr(predictions, "flatten") and len(predictions) > 0:
            preview_len = min(10, len(predictions))
            print(f"[RUN] First {preview_len}: {predictions[:preview_len]}")

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
        Number of trials to run.
    """
    _print_separator("Starting HYPERPARAMETER TUNING pipeline")
    if n_trials:
        print(f"[RUN] Trials override: {n_trials}")

    merged_cfg = merge_configs(experiment_path, pipeline_path, mode="tune")

    if n_trials:
        if "tuning" not in merged_cfg:
            merged_cfg.tuning = {}
        merged_cfg.tuning.n_trials = n_trials

        for step in merged_cfg.pipeline.steps:
            if step.get("type") == "tuner":
                step.n_trials = n_trials
                print(f"[CONFIG] Override: n_trials={n_trials}")
                break

    temp_config = ".temp_merged_tune.yaml"
    save_merged_config(merged_cfg, temp_config)

    try:
        pipeline = FlexiblePipeline(temp_config)
        context = pipeline.run_exp()

        _print_separator("Tuning COMPLETE")

        for key in context:
            if "_best_params" in key:
                print(f"[RUN] Best parameters ({key}):")
                for param, value in context[key].items():
                    print(f"  - {param}: {value}")

        for key in context:
            if "_study" in key:
                study = context[key]
                print(f"[RUN] Best metric value: {study.best_value:.4f}")
                break

    finally:
        Path(temp_config).unlink(missing_ok=True)


def run_generate_configs(
    train_config: str,
    output_dir: str = "mlproject/configs/generated",
    alias: str = "latest",
    config_type: str = "all",
) -> None:
    """Generate eval/serve configs from training config.

    Parameters
    ----------
    train_config : str
        Path to training experiment YAML.
    output_dir : str
        Output directory for generated configs.
    alias : str
        MLflow model alias.
    config_type : str
        Type of config to generate (eval/serve/all).
    """
    _print_separator("GENERATING CONFIGS")
    print(f"[RUN] Source: {train_config}")
    print(f"[RUN] Output: {output_dir}")

    generator = ConfigGenerator(train_config)
    base_name = Path(train_config).stem

    if config_type == "all":
        paths = generator.generate_all(output_dir, alias)
        print("\nGenerated configs:")
        print(f"  - Eval:  {paths['eval']}")
        print(f"  - Serve: {paths['serve']}")
    elif config_type == "eval":
        out_path = str(Path(output_dir) / f"{base_name}_eval.yaml")
        generator.generate_eval_config(alias=alias, output_path=out_path)
        print(f"Generated: {out_path}")
    else:
        out_path = str(Path(output_dir) / f"{base_name}_serve.yaml")
        generator.generate_serve_config(alias=alias, output_path=out_path)
        print(f"Generated: {out_path}")


def _setup_train_parser(subparsers: Any) -> None:
    """Setup train subcommand parser."""
    parser = subparsers.add_parser("train", help="Run training pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")


def _setup_eval_parser(subparsers: Any) -> None:
    """Setup eval subcommand parser."""
    parser = subparsers.add_parser("eval", help="Run evaluation pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--alias", "-a", default="latest", help="Model alias")


def _setup_serve_parser(subparsers: Any) -> None:
    """Setup serve subcommand parser."""
    parser = subparsers.add_parser("serve", help="Run serve pipeline")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--input", "-i", help="Input CSV file")
    parser.add_argument("--alias", "-a", default="latest", help="Model alias")
    parser.add_argument("--time_point", "-t", default="now", help="Timestamp for Feast")


def _setup_tune_parser(subparsers: Any) -> None:
    """Setup tune subcommand parser."""
    parser = subparsers.add_parser("tune", help="Run hyperparameter tuning")
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment config YAML"
    )
    parser.add_argument("--pipeline", "-p", help="Pipeline config YAML")
    parser.add_argument("--trials", "-n", type=int, help="Number of tuning trials")


def _setup_generate_parser(subparsers: Any) -> None:
    """Setup generate subcommand parser."""
    parser = subparsers.add_parser("generate", help="Generate eval/serve configs")
    parser.add_argument(
        "--train-config", "-t", required=True, help="Training config YAML"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="mlproject/configs/generated",
        help="Output directory",
    )
    parser.add_argument("--alias", "-a", default="latest", help="MLflow model alias")
    parser.add_argument(
        "--type",
        choices=["eval", "serve", "all"],
        default="all",
        help="Type of config to generate",
    )


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run ML pipelines (train/eval/serve/tune/generate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline type")

    _setup_train_parser(subparsers)
    _setup_eval_parser(subparsers)
    _setup_serve_parser(subparsers)
    _setup_tune_parser(subparsers)
    _setup_generate_parser(subparsers)

    args = parser.parse_args()

    try:
        if args.command == "train":
            run_training(args.experiment, args.pipeline)
        elif args.command == "eval":
            run_eval(args.experiment, args.pipeline, args.alias)
        elif args.command == "serve":
            run_serve(
                args.experiment,
                args.pipeline,
                args.input,
                args.alias,
                args.time_point,
            )
        elif args.command == "tune":
            run_tune(args.experiment, args.pipeline, args.trials)
        elif args.command == "generate":
            run_generate_configs(
                args.train_config,
                args.output_dir,
                args.alias,
                args.type,
            )
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
