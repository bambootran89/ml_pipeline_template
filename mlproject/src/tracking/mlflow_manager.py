"""
High-level MLflowManager orchestrator.

Provides a unified interface to manage MLflow operations:
- Experiment setup and restoration
- Run lifecycle management
- Logging parameters, metrics, artifacts, and configuration
- Model logging and loading via PyFunc
- Model registration in MLflow Registry
"""

import os
from typing import Any, Optional

import mlflow

from .config_logger import ConfigLogger
from .experiment_manager import ExperimentManager
from .model_logger import ModelLogger
from .registry_manager import RegistryManager
from .run_manager import RunManager


class MLflowManager:
    """
    High-level controller for MLflow operations.

    Attributes:
        cfg (Any): Full project configuration object.
        mlflow_cfg (dict): MLflow-specific configuration extracted from cfg.
        enabled (bool): Indicates if MLflow tracking is enabled.
        experiment (Optional[mlflow.entities.Experiment]): Active MLflow experiment.
        run_manager (RunManager): Manages MLflow runs.
        experiment_manager (ExperimentManager): Manages experiments.
        model_logger (ModelLogger): Handles model logging/loading.
        registry_manager (RegistryManager): Handles registry operations.
        config_logger (ConfigLogger): Handles params, metrics, and config logging.
    """

    def __init__(self, cfg: Any):
        """
        Initialize MLflowManager and set up the experiment.

        Args:
            cfg (Any): Full project configuration containing an optional 'mlflow' block.
        """
        self.cfg = cfg
        self.mlflow_cfg = cfg.get("mlflow", {})
        self.enabled = self.mlflow_cfg.get("enabled", False)

        self.experiment_manager = ExperimentManager(self.mlflow_cfg, self.enabled)
        self.run_manager = RunManager(self.mlflow_cfg, self.enabled)
        self.model_logger = ModelLogger(self.mlflow_cfg, self.enabled)
        self.registry_manager = RegistryManager(self.mlflow_cfg, self.enabled)
        self.config_logger = ConfigLogger()
        if not self.enabled:
            return
        mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns"))
        self.experiment = self.experiment_manager.setup_experiment()

    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """
        Start an MLflow run.

        Args:
            run_name (Optional[str]):
                Name for the run. Defaults to a timestamped name if None.
            nested (bool): If True, start a nested run. Defaults to False.
        Returns:
            mlflow.ActiveRun: Context manager for the run.
        """
        return self.run_manager.start_run(run_name, nested=nested)

    def log_params(self, params: dict):
        """
        Log parameters to MLflow.

        Args:
            params (dict): Parameters to log.
        """
        if not self.enabled:
            return
        self.config_logger.log_params(params)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics (dict): Metrics to log.
            step (Optional[int]): Step number (e.g., epoch). Defaults to None.
        """
        if not self.enabled:
            return
        self.config_logger.log_metrics(metrics, step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to the current MLflow run.

        Args:
            local_path (str): Path to the local file.
            artifact_path (Optional[str]): Destination path within the MLflow run.
        """
        if not self.enabled:
            return
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)

    def log_config(self):
        """
        Log the full project configuration to MLflow.
        """
        if not self.enabled:
            return
        self.config_logger.log_config(self.cfg)

    def log_model(self, *args, **kwargs):
        """
        Log a model using the ModelLogger.

        Args:
            *args, **kwargs: Forwarded to ModelLogger.log_model.
        """
        return self.model_logger.log_model(*args, **kwargs)

    def load_model(self, model_uri: str):
        """
        Load a model from MLflow.

        Args:
            model_uri (str): MLflow model URI.

        Returns:
            mlflow.pyfunc.PyFuncModel: Loaded PyFunc model.
        """
        return self.model_logger.load_model(model_uri)

    def register_model(self, *args, **kwargs):
        """
        Register a model in MLflow Model Registry.

        Args:
            *args, **kwargs: Forwarded to RegistryManager.register_model.

        Returns:
            mlflow.entities.ModelVersion: Registered model version.
        """
        return self.registry_manager.register_model(*args, **kwargs)
