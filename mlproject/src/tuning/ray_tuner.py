"""
Ray Tune integration for distributed hyperparameter tuning.

Features:
- Distributed hyperparameter search with Ray Tune
- ASHA early-stopping scheduler
- Automatic conversion from project search space to Ray Tune space
"""

from copy import deepcopy
from typing import Any, Dict, Optional

from omegaconf import DictConfig

from mlproject.src.cv.cv_pipeline import CrossValidationPipeline
from mlproject.src.cv.splitter import TimeSeriesSplitter
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.base_tuner import BaseTuner

try:
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


class DummyMLflowManager(MLflowManager):
    """A disabled MLflow manager used when MLflowManager is None."""

    def __init__(self) -> None:
        """Create an MLflow stub with logging disabled."""
        super().__init__(cfg=None)

    @property
    def enabled(self) -> bool:
        """Return False to disable logging."""
        return False


class RayTuner(BaseTuner):
    """
    Ray Tune tuner for distributed hyperparameter optimization.
    """

    def __init__(
        self,
        cfg: DictConfig,
        splitter: TimeSeriesSplitter,
        mlflow_manager: Optional[MLflowManager] = None,
        metric_name: str = "mae_mean",
        direction: str = "minimize",
    ) -> None:
        """
        Initialize RayTuner.

        Raises:
            ImportError: If Ray Tune is not installed.
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is not available. Install via: pip install ray[tune]"
            )

        super().__init__(cfg, splitter, mlflow_manager, metric_name, direction)

    # ----------------------------------------------------------------------
    # Helper to reduce locals in tune() — fixes R0914
    # ----------------------------------------------------------------------
    def _prepare_tune_components(
        self,
        model_name: str,
        n_trials: int,
    ) -> tuple[Dict[str, Any], ASHAScheduler, str]:
        """
        Prepare Ray Tune search space, scheduler, and optimization mode.

        Args:
            model_name: Name of the ML model.
            n_trials: Maximum number of ASHA iterations.

        Returns:
            Tuple containing:
            - search_space: Ray Tune search space
            - scheduler: ASHAScheduler instance
            - mode: "min" or "max"
        """
        search_space = self._convert_search_space(model_name)
        scheduler = ASHAScheduler(
            max_t=n_trials,
            grace_period=5,
            reduction_factor=2,
        )
        mode = "min" if self.direction == "minimize" else "max"
        return search_space, scheduler, mode

    def _convert_search_space(self, model_name: str) -> Dict[str, Any]:
        """
        Convert project search space into Ray Tune search space.
        """
        search_space_def = self.get_search_space(model_name)
        ray_space: Dict[str, Any] = {}

        for param_name, param_config in search_space_def.items():
            ptype = param_config["type"]
            prange = param_config["range"]

            if ptype == "int":
                ray_space[param_name] = tune.randint(prange[0], prange[1] + 1)
            elif ptype == "float":
                if param_config.get("log", False):
                    ray_space[param_name] = tune.loguniform(prange[0], prange[1])
                else:
                    ray_space[param_name] = tune.uniform(prange[0], prange[1])
            elif ptype == "categorical":
                ray_space[param_name] = tune.choice(prange)

        return ray_space

    def objective(self, trial: Dict[str, Any]) -> float:
        """
        Objective executed for each Ray Tune trial.

        Args:
            trial: Hyperparameters sampled by Ray Tune.

        Returns:
            float: Optimization metric value.
        """
        mlflow_mgr = self.mlflow_manager or DummyMLflowManager()
        model_name = self.cfg.experiment.model.lower()

        trial_cfg = deepcopy(self.cfg)
        trial_cfg.experiment.hyperparams.update(trial)

        cv_pipeline = CrossValidationPipeline(
            trial_cfg,
            self.splitter,
            mlflow_mgr,
        )

        approach = {"model": model_name, "hyperparams": trial}

        data = cv_pipeline.preprocess()
        metrics = cv_pipeline.run_cv(approach, data)

        tune.report(**metrics)
        return float(metrics[self.metric_name])

    def _format_best_result(
        self, metric_value: float, best_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a small dictionary containing formatted result strings.

        This reduces the number of locals inside tune() and fixes R0914.
        """
        lines = {
            "header": "\n" + "=" * 60,
            "footer": "=" * 60 + "\n",
            "summary": f"Best {self.metric_name}: {metric_value:.6f}",
            "params": [f"{k:20} = {v}" for k, v in best_params.items()],
        }
        return lines

    def tune(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        num_samples: int = 10,
        max_concurrent: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute Ray Tune hyperparameter search.

        Returns:
            dict with best_params, best_value, and analysis.
        """
        model_name = self.cfg.experiment.model.lower()

        # R0914 solved by moving 3 locals + using formatted dict
        search_space, scheduler, mode = self._prepare_tune_components(
            model_name, n_trials
        )

        analysis = tune.run(
            self.objective,
            config=search_space,
            metric=self.metric_name,
            mode=mode,
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent,
            verbose=1,
            **kwargs,
        )

        best_trial = analysis.get_best_trial(self.metric_name, mode, "last")
        best_info = {
            "params": best_trial.config,
            "value": float(best_trial.last_result[self.metric_name]),
        }

        # Reduced locals → fixes R0914
        print("\n" + "=" * 60)
        print("  RAY TUNE COMPLETED")
        print("=" * 60)
        print(f"Best {self.metric_name}: {best_info['value']:.6f}")
        print("Best hyperparameters:")
        for k, v in best_info["params"].items():
            print(f"  {k:20} = {v}")
        print("=" * 60 + "\n")

        return {
            "best_params": best_info["params"],
            "best_value": best_info["value"],
            "analysis": analysis,
        }
