"""
Optuna-based hyperparameter tuning for time-series cross-validation.

This tuner supports:
- Automatic search-space loading via SearchSpaceRegistry
- MLflow integration for experiment tracking
- Pruning support
- Parallel optimization execution
"""

from copy import deepcopy
from typing import Any, Dict, Optional

import optuna
from omegaconf import DictConfig
from optuna.integration.mlflow import MLflowCallback

from mlproject.src.cv.cv_pipeline import CrossValidationPipeline
from mlproject.src.cv.splitter import TimeSeriesSplitter
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.base_tuner import BaseTuner


class OptunaTuner(BaseTuner):
    """Optuna tuner for time-series forecasting models."""

    def __init__(
        self,
        cfg: DictConfig,
        splitter: TimeSeriesSplitter,
        mlflow_manager: Optional[MLflowManager] = None,
        metric_name: str = "mae_mean",
        direction: str = "minimize",
    ):
        """Initialize the Optuna tuner."""
        super().__init__(cfg, splitter, mlflow_manager, metric_name, direction)

        # Guarantee MLflowManager instance for mypy
        if mlflow_manager is None:
            self.mlflow_manager = MLflowManager(cfg)
            self.mlflow_manager.enabled = False
        else:
            self.mlflow_manager = mlflow_manager

        # CV pipeline used inside the objective function
        self.cv_pipeline = CrossValidationPipeline(cfg, splitter, self.mlflow_manager)

    def _suggest_params(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters from the search space."""
        search_space = self.get_search_space(model_name)
        params: Dict[str, Any] = {}

        for name, spec in search_space.items():
            param_type = spec["type"]
            param_range = spec["range"]

            if param_type == "int":
                step = spec.get("step", 1)
                params[name] = trial.suggest_int(
                    name, param_range[0], param_range[1], step=step
                )

            elif param_type == "float":
                log = spec.get("log", False)
                params[name] = trial.suggest_float(
                    name, param_range[0], param_range[1], log=log
                )

            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, param_range)

            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective evaluated for each trial."""
        model_name = self.cfg.experiment.model.lower()

        # 1. Suggest hyperparameters
        hyperparams = self._suggest_params(trial, model_name)

        # Include fixed params
        fixed = dict(self.cfg.experiment.hyperparams)
        for key in ["input_chunk_length", "output_chunk_length"]:
            if key in fixed:
                hyperparams[key] = fixed[key]

        # 2. Build trial-specific config
        trial_cfg = deepcopy(self.cfg)
        trial_cfg.experiment.hyperparams.update(hyperparams)

        self.cv_pipeline.cfg = trial_cfg

        # 3. Run cross validation
        approach = {"model": model_name, "hyperparams": hyperparams}
        data = self.cv_pipeline.preprocess()
        metrics = self.cv_pipeline.run_cv(approach, data)

        return metrics[self.metric_name]

    def tune(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the Optuna hyperparameter search."""
        n_jobs = kwargs.get("n_jobs", 1)
        show_progress = kwargs.get("show_progress", True)

        study = optuna.create_study(
            direction=self.direction,
            study_name=f"{self.cfg.experiment.name}_tuning",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        callbacks = []
        if self.mlflow_manager and getattr(self.mlflow_manager, "enabled", False):
            mlflow_cb = MLflowCallback(
                tracking_uri=self.mlflow_manager.mlflow_cfg.get(
                    "tracking_uri", "mlruns"
                ),
                metric_name=self.metric_name,
            )
            callbacks.append(mlflow_cb)

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=callbacks or None,
            show_progress_bar=show_progress,
        )

        print("\n" + "=" * 60)
        print("  OPTUNA TUNING COMPLETED")
        print("=" * 60)
        print(f"Best {self.metric_name}: {study.best_value:.6f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key:20} = {value}")
        print(f"Total trials: {len(study.trials)}")
        print("=" * 60 + "\n")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "study": study,
        }
