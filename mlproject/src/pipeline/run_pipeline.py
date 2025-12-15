"""
Extended command-line interface (CLI) supporting Training, Evaluation,
Testing, Cross-Validation (CV), and Hyperparameter Tuning.

Usage examples:
    # Standard modes
    python -m mlproject.src.pipeline.run_pipeline train --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline eval --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline test --config path.yaml \
        --input file.csv

    # Additional modes
    python -m mlproject.src.pipeline.run_pipeline cv --config path.yaml
    python -m mlproject.src.pipeline.run_pipeline tune --config path.yaml
"""

import argparse
from typing import Any, Dict, cast

import pandas as pd
from omegaconf import OmegaConf

from mlproject.src.datamodule.base_splitter import BaseSplitter
from mlproject.src.datamodule.ts_splitter import TimeSeriesFoldSplitter
from mlproject.src.pipeline.cv_pipeline import CrossValidationPipeline
from mlproject.src.pipeline.engines.tuning_pipeline import TuningPipeline

# === moved all imports to top to fix pylint C0415 ===
from mlproject.src.pipeline.eval_pipeline import EvalPipeline
from mlproject.src.pipeline.serve_pipeline import TestPipeline
from mlproject.src.pipeline.training_pipeline import TrainingPipeline
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader

# ====================================================


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["OMP_NUM_THREADS"] = "1"


def run_training(cfg_path: str) -> None:
    """Execute the full training workflow."""
    pipeline = TrainingPipeline(cfg_path)
    pipeline.run()


def run_evaluation(cfg_path: str) -> None:
    """Execute evaluation-only workflow."""
    pipeline = EvalPipeline(cfg_path)
    pipeline.run()


def run_testing(cfg_path: str, input_path: str) -> None:
    """
    Execute serving-time inference on a provided CSV file.
    """
    if not input_path:
        raise ValueError("Test mode requires --input <file.csv>")

    pipeline = TestPipeline(cfg_path)

    raw_df = pd.read_csv(input_path)
    assert "date" in raw_df.columns, "Input CSV must contain a 'date' column."
    raw_df.date = pd.to_datetime(raw_df.date)
    raw_df = raw_df.set_index("date")

    pipeline.run(raw_df)


def run_cross_validation(cfg_path: str) -> None:
    """
    Execute time-series cross-validation using ExpandingWindowSplitter.
    """
    cfg = ConfigLoader.load(cfg_path)
    mlflow_manager = MLflowManager(cfg)

    cfg_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    splitter: BaseSplitter
    eval_type = cfg.get("data", {}).get("type", "timeseries")
    if eval_type == "timeseries":
        splitter = cast(
            BaseSplitter,
            TimeSeriesFoldSplitter(
                cfg_dict,
                n_splits=cfg.get("tuning", {}).get("n_splits", 3),
            ),
        )
    else:
        splitter = BaseSplitter(
            cfg_dict,
            n_splits=cfg.get("tuning", {}).get("n_splits", 3),
        )

    cv_pipeline = CrossValidationPipeline(cfg, splitter, mlflow_manager)

    approach = {
        "model": cfg.experiment.model,
        "hyperparams": dict(cfg.experiment.hyperparams),
    }

    cv_pipeline.run_cv(approach)


def run_tuning(cfg_path: str) -> None:
    """
    Execute hyperparameter tuning workflow.
    """
    tuning_pipeline = TuningPipeline(cfg_path)
    tuning_pipeline.run()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build unified CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified entrypoint for Training, Evaluation, Testing, "
            "Cross-Validation, and Hyperparameter Tuning."
        )
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "test", "cv", "tune"],
        help="Which workflow pipeline to execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="CSV path to use only for test mode.",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.config)
    elif args.mode == "eval":
        run_evaluation(args.config)
    elif args.mode == "test":
        run_testing(args.config, args.input)
    elif args.mode == "cv":
        run_cross_validation(args.config)
    elif args.mode == "tune":
        run_tuning(args.config)
    else:
        raise ValueError(f"Unknown mode '{args.mode}'.")


def main_run(mode: str, cfg_path: str = "", input_path: str = "") -> None:
    """Programmatic entrypoint."""
    if mode == "train":
        run_training(cfg_path)
    elif mode == "eval":
        run_evaluation(cfg_path)
    elif mode == "test":
        run_testing(cfg_path, input_path)
    elif mode == "cv":
        run_cross_validation(cfg_path)
    elif mode == "tune":
        run_tuning(cfg_path)
    else:
        raise ValueError(f"Unknown mode '{mode}'.")


if __name__ == "__main__":
    main()
