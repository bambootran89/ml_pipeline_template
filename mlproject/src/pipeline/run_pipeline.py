import argparse

import pandas as pd

from mlproject.src.pipeline.eval_pipeline import EvalPipeline
from mlproject.src.pipeline.serve_pipeline import TestPipeline
from mlproject.src.pipeline.training_pipeline import TrainingPipeline


def run_training(cfg_path: str) -> None:
    """Run the full training pipeline."""
    pipeline = TrainingPipeline(cfg_path)
    pipeline.run()  # executes preprocessing and training


def run_evaluation(cfg_path: str) -> None:
    """Run the evaluation-only pipeline."""
    pipeline = EvalPipeline(cfg_path)
    pipeline.run()  # executes evaluation


def run_testing(cfg_path: str, input_path: str) -> None:
    """Run inference pipeline on a CSV file."""
    if not input_path:
        raise ValueError("Test mode requires --input <file.csv>")

    pipeline = TestPipeline(cfg_path)
    raw_df = pd.read_csv(input_path)
    assert "date" in raw_df.columns
    raw_df = raw_df.set_index("date")

    # run pipeline without assigning return
    pipeline.run(raw_df)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for Training / Evaluation / Testing pipelines."
    )

    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "test"],
        help="Which pipeline to run.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to config YAML. If omitted, default experiment config is used.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="CSV file path for test inference mode.",
    )

    return parser


def main() -> None:
    """CLI entrypoint for running train/eval/test pipelines."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args.config)
    elif args.mode == "eval":
        run_evaluation(args.config)
    elif args.mode == "test":
        run_testing(args.config, args.input)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


def main_run(mode: str, cfg_path: str = "", input_path: str = "") -> None:
    """
    Programmatic entrypoint for running pipelines.

    Args:
        mode (str): "train", "eval", or "test"
        cfg_path (str): path to config YAML
        input_path (str): path to CSV for test mode
    """
    if mode == "train":
        run_training(cfg_path)
    elif mode == "eval":
        run_evaluation(cfg_path)
    elif mode == "test":
        run_testing(cfg_path, input_path)
    else:
        raise ValueError(f"Unknown mode {mode}")


if __name__ == "__main__":
    main()
