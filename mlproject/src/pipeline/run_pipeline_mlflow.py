"""
CLI entrypoint cho training/eval pipelines với MLflow.
"""
import argparse

import pandas as pd

from mlproject.src.pipeline.eval_pipeline_mlflow import EvalPipelineMLflow
from mlproject.src.pipeline.serve_pipeline import TestPipeline
from mlproject.src.pipeline.training_pipeline_mlflow import TrainingPipelineMLflow


def run_training(cfg_path: str) -> None:
    """
    Run training pipeline với MLflow tracking.

    Args:
        cfg_path: Path to config YAML
    """
    pipeline = TrainingPipelineMLflow(cfg_path)
    pipeline.run()


def run_evaluation(
    cfg_path: str,
    model_name: str = "",
    model_version: str = "latest",
) -> None:
    """
    Run evaluation pipeline load model từ MLflow.

    Args:
        cfg_path: Path to config YAML
        model_name: Model name trong registry
        model_version: Model version
    """
    pipeline = EvalPipelineMLflow(cfg_path, model_name, model_version)
    pipeline.run()


def run_testing(cfg_path: str, input_path: str) -> None:
    """
    Run inference pipeline trên CSV file.

    Args:
        cfg_path: Path to config YAML
        input_path: Path to CSV file
    """
    if not input_path:
        raise ValueError("Test mode requires --input <file.csv>")

    pipeline = TestPipeline(cfg_path)
    raw_df = pd.read_csv(input_path)
    assert "date" in raw_df.columns
    raw_df = raw_df.set_index("date")

    pipeline.run(raw_df)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="ML Pipeline với MLflow tracking and registry"
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "test"],
        help="Pipeline mode: train/eval/test",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config YAML",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="CSV file path cho test mode",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name trong MLflow registry (cho eval mode)",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Model version: 'latest', '1', '2', ... (cho eval mode)",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.config)

    elif args.mode == "eval":
        run_evaluation(args.config, args.model_name, args.model_version)

    elif args.mode == "test":
        run_testing(args.config, args.input)

    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
