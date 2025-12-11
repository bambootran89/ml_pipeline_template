"""
Example script for running cross-validation using existing hyperparameters.

Usage:
    python scripts/run_cv.py --config mlproject/configs/experiments/etth2.yaml
"""

import argparse

from mlproject.src.datamodule.splitter import ExpandingWindowSplitter
from mlproject.src.eval.cv_reporter import CVEvaluator
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.pipeline.cv_pipeline import CrossValidationPipeline
from mlproject.src.tracking.mlflow_manager import MLflowManager


def main() -> None:
    """Execute the cross-validation pipeline."""
    parser = argparse.ArgumentParser(description="Run Cross-Validation")
    parser.add_argument(
        "--config",
        type=str,
        default="mlproject/configs/experiments/etth2.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=20,
        help="Test set size for each fold.",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = ConfigLoader.load(args.config)

    # Initialize components
    splitter = ExpandingWindowSplitter(
        n_splits=args.n_splits,
        test_size=args.test_size,
    )
    mlflow_manager = MLflowManager(cfg)
    cv_pipeline = CrossValidationPipeline(cfg, splitter, mlflow_manager)

    # Preprocess dataset
    print("Preprocessing data...")
    data = cv_pipeline.preprocess()

    # Run cross-validation
    print(f"\nRunning {args.n_splits}-fold cross-validation...")
    approach = {
        "model": cfg.experiment.model,
        "hyperparams": dict(cfg.experiment.hyperparams),
    }

    metrics = cv_pipeline.run_cv(approach, data)

    # Summary output
    CVEvaluator()
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for key, val in sorted(metrics.items()):
        print(f"{key:20} = {val:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
