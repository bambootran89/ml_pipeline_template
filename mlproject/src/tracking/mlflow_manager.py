"""
High-level MLflow Manager orchestrating:

- Experiment setup
- Autologging
- run lifecycle (delegated to RunManager)
- logging params/metrics
- config logging
- forwarding model logging to ModelLogger
- forwarding registry ops to RegistryManager
"""

import os
from typing import Any, Dict, Optional

import mlflow
from omegaconf import DictConfig, OmegaConf

from .model_logger import ModelLogger
from .registry_manager import RegistryManager
from .run_manager import RunManager


class MLflowManager:
    """
    Central MLflow controller.

    Handles:
    - Experiment setup
    - Autologging config
    - Run lifecycle
    - Parameter & metric logging
    - Model logging (via ModelLogger)
    - Registry (via RegistryManager)
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Full project configuration.
        """
        self.cfg = cfg
        self.mlflow_cfg = cfg.get("mlflow", {})
        self.enabled = self.mlflow_cfg.get("enabled", False)

        if not self.enabled:
            return

        mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns"))

        self._setup_experiment()
        self._setup_autologging()

        # module split
        self.run_manager = RunManager(self.mlflow_cfg)
        self.model_logger = ModelLogger(self.mlflow_cfg, self.enabled)
        self.registry_manager = RegistryManager(self.mlflow_cfg, self.enabled)

    # -------------------------------
    # Experiment management
    # -------------------------------
    def _setup_experiment(self):
        """Create or restore the MLflow experiment."""
        exp_name = self.mlflow_cfg.get("experiment_name", "Default_Experiment")

        try:
            self.experiment = mlflow.set_experiment(exp_name)

        except mlflow.exceptions.MlflowException as e:
            if "deleted experiment" not in str(e).lower():
                raise e

            print(f"[MLflow] Experiment '{exp_name}' deleted → restoring…")

            client = mlflow.MlflowClient()
            deleted = client.search_experiments(
                view_type=mlflow.entities.ViewType.DELETED_ONLY
            )

            for exp in deleted:
                if exp.name == exp_name:
                    client.restore_experiment(exp.experiment_id)
                    self.experiment = mlflow.set_experiment(exp_name)
                    return

            new_name = f"{exp_name}_v2"
            print(f"[MLflow] Could not restore → creating {new_name}")
            self.experiment = mlflow.set_experiment(new_name)

    def _setup_autologging(self):
        """Enable autologging but disable model autolog for custom pyfunc logs."""
        if self.mlflow_cfg.get("autolog", True):
            mlflow.autolog(log_models=False, exclusive=False)
            print("[MLflow] Autologging enabled.")

    # -------------------------------
    # Delegated run lifecycle
    # -------------------------------
    def start_run(self, run_name: Optional[str] = None):
        """
        Forward to RunManager for context manager.

        Usage:
            with mlflow_manager.start_run("train"):
                ...
        """
        return self.run_manager.start_run(run_name)

    # -------------------------------
    # Logging utilities
    # -------------------------------
    def log_params(self, params: Dict[str, Any]):
        """Flatten dictionary & log params."""
        if not self.enabled:
            return
        mlflow.log_params(self._flatten_dict(params))

    def log_metrics(self, metrics: Dict[str, float], step=None):
        """Log metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log single artifact file."""
        if not self.enabled:
            return
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)

    def _log_config(self):
        """Log full Hydra/OmegaConf config as YAML."""
        cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow.log_dict(cfg_dict, "config/full_config.yaml")

    # -------------------------------
    # Forwarding to submodules
    # -------------------------------
    def log_model(self, *args, **kwargs):
        return self.model_logger.log_model(*args, **kwargs)

    def load_model(self, model_uri: str):
        return self.model_logger.load_model(model_uri)

    def register_model(self, *args, **kwargs):
        return self.registry_manager.register_model(*args, **kwargs)

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _flatten_dict(self, d, parent_key="", sep="."):
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
