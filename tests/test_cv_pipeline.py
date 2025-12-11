"""
Integration test for CV pipeline.

Run:
    pytest tests/test_cv_pipeline.py -v -s
"""

import pytest

from mlproject.src.cv.cv_pipeline import CrossValidationPipeline
from mlproject.src.cv.splitter import ExpandingWindowSplitter
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.tracking.mlflow_manager import MLflowManager


@pytest.fixture
def cv_pipeline():
    """Create CV pipeline với test config."""
    cfg = ConfigLoader.load("mlproject/configs/experiments/etth2.yaml")

    # Disable MLflow cho test
    cfg.mlflow.enabled = False

    # Small n_epochs cho fast test
    cfg.experiment.hyperparams.n_epochs = 2

    splitter = ExpandingWindowSplitter(n_splits=2, test_size=10)
    mlflow_manager = MLflowManager(cfg)

    return CrossValidationPipeline(cfg, splitter, mlflow_manager)


def test_cv_pipeline_smoke(cv_pipeline):
    """Smoke test: CV pipeline runs without errors."""
    # Preprocess
    data = cv_pipeline.preprocess()
    assert len(data) > 0

    # Run CV
    approach = {
        "model": cv_pipeline.cfg.experiment.model,
        "hyperparams": dict(cv_pipeline.cfg.experiment.hyperparams),
    }

    metrics = cv_pipeline.run_cv(approach, data)

    # Check aggregated metrics
    assert "mae_mean" in metrics
    assert "mae_std" in metrics
    assert metrics["mae_mean"] > 0


def test_cv_metrics_format(cv_pipeline):
    """Test that aggregated metrics have correct format."""
    data = cv_pipeline.preprocess()

    approach = {
        "model": cv_pipeline.cfg.experiment.model,
        "hyperparams": dict(cv_pipeline.cfg.experiment.hyperparams),
    }

    metrics = cv_pipeline.run_cv(approach, data)

    # Should have mean, std, min, max cho mỗi metric
    for base_metric in ["mae", "mse", "rmse"]:
        assert f"{base_metric}_mean" in metrics
        assert f"{base_metric}_std" in metrics
