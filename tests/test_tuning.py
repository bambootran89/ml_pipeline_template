"""
Test tuning components.

Run:
    pytest tests/test_tuning.py -v -s
"""

import pytest

from mlproject.src.datamodule.ts_splitter import TimeSeriesFoldSplitter
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna_tuner import OptunaTuner
from mlproject.src.tuning.search_space import SearchSpaceRegistry


class TestSearchSpace:
    """Test search space registry."""

    def test_get_xgboost_space(self):
        """Test getting XGBoost search space."""
        space = SearchSpaceRegistry.get("xgboost")

        assert "n_estimators" in space
        assert "max_depth" in space
        assert "learning_rate" in space

        # Check types
        assert space["n_estimators"]["type"] == "int"
        assert space["learning_rate"]["type"] == "float"
        assert space["learning_rate"]["log"] is True

    def test_get_nlinear_space(self):
        """Test getting NLinear search space."""
        space = SearchSpaceRegistry.get("nlinear")

        assert "hidden" in space
        assert "lr" in space
        assert "batch_size" in space

        # Check categorical
        assert space["batch_size"]["type"] == "categorical"
        assert space["batch_size"]["range"] == [16, 32, 64]

    def test_list_models(self):
        """Test listing available models."""
        models = SearchSpaceRegistry.list_models()

        assert "xgboost" in models
        assert "nlinear" in models
        assert "tft" in models

    def test_register_custom(self):
        """Test registering custom search space."""
        custom_space = {
            "param1": {"type": "int", "range": [1, 10]},
            "param2": {"type": "float", "range": [0.1, 1.0]},
        }

        SearchSpaceRegistry.register("custom_model", custom_space)

        retrieved = SearchSpaceRegistry.get("custom_model")
        assert retrieved == custom_space


@pytest.mark.slow
def test_optuna_tuner_smoke():
    """
    Smoke test for Optuna tuner.

    Mark as slow because it runs actual tuning.
    Run: pytest tests/test_tuning.py::test_optuna_tuner_smoke -v -s
    """
    cfg = ConfigLoader.load("mlproject/configs/experiments/etth2.yaml")

    # Disable MLflow
    cfg.mlflow.enabled = False

    # Fast test settings
    cfg.experiment.hyperparams.n_epochs = 1

    splitter = TimeSeriesFoldSplitter(
        cfg,
        n_splits=2,
    )
    mlflow_manager = MLflowManager(cfg)

    tuner = OptunaTuner(
        cfg=cfg,
        splitter=splitter,
        mlflow_manager=mlflow_manager,
        metric_name="mae_mean",
        direction="minimize",
    )

    # Run 2 trials only
    result = tuner.tune(n_trials=2, show_progress=False)

    assert "best_params" in result
    assert "best_value" in result
    assert "study" in result

    # Check best_params có hyperparameters
    assert len(result["best_params"]) > 0
