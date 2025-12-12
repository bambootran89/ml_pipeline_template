"""
Example script for running cross-validation using existing hyperparameters.

Usage:
    python scripts/run_cv.py --config mlproject/configs/experiments/etth2.yaml
"""

import argparse
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from mlproject.src.datamodule.splitter import TimeSeriesFoldSplitter
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.pipeline.cv_pipeline import CrossValidationPipeline
from mlproject.src.tracking.mlflow_manager import MLflowManager


def main() -> None:
    """Execute the cross-validation pipeline.

    This script loads experiment configuration, initializes the time-series
    fold splitter, MLflow tracking manager, and cross-validation pipeline,
    and then executes cross-validation using predefined hyperparameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If configuration values are missing or invalid.
    """
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

    # Load configuration as DictConfig
    cfg: DictConfig = ConfigLoader.load(args.config)

    # Convert DictConfig → dict for components requiring plain dict (mypy-safe)
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(
        cfg, resolve=True
    )  # type: ignore[assignment]

    # Initialize components
    splitter = TimeSeriesFoldSplitter(
        cfg_dict,  # <-- FIX: mypy now recognizes correct type
        n_splits=args.n_splits,
    )

    mlflow_manager = MLflowManager(cfg)
    cv_pipeline = CrossValidationPipeline(cfg, splitter, mlflow_manager)

    # Run cross-validation
    print(f"\nRunning {args.n_splits}-fold cross-validation...")
    approach: Dict[str, Any] = {
        "model": cfg.experiment.model,
        "hyperparams": dict(cfg.experiment.hyperparams),
    }

    metrics = cv_pipeline.run_cv(approach)

    # Summary output
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for key, val in sorted(metrics.items()):
        print(f"{key:20} = {val:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
