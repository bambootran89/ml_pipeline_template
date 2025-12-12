import os
from contextlib import contextmanager
from typing import Any, ContextManager, Dict, Iterator, Optional

import mlflow

from .config_logger import ConfigLogger
from .experiment_manager import ExperimentManager
from .model_logger import ModelLogger
from .registry_manager import RegistryManager
from .run_manager import RunManager


class MLflowManager:
    """High-level MLflow controller.

    Wraps all MLflow operations via sub-managers:
    ExperimentManager, RunManager, ModelLogger, RegistryManager, ConfigLogger.

    Safe: when MLflow is disabled, all methods become silent no-ops.
    """

    def __init__(self, cfg: Any) -> None:
        """Initialize MLflowManager.

        Args:
            cfg: Global configuration object (dict or DictConfig).
        """
        self.cfg = cfg
        self.mlflow_cfg: Dict[str, Any] = cfg.get("mlflow", {}) if cfg else {}
        self.enabled: bool = bool(self.mlflow_cfg.get("enabled", False))

        self.experiment_manager = ExperimentManager(self.mlflow_cfg, self.enabled)
        self.run_manager = RunManager(self.mlflow_cfg, self.enabled)
        self.model_logger = ModelLogger(self.mlflow_cfg, self.enabled)
        self.registry_manager = RegistryManager(self.mlflow_cfg, self.enabled)
        self.config_logger = ConfigLogger()

        if not self.enabled:
            self.experiment = None
            return

        mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns"))
        self.experiment = self.experiment_manager.setup_experiment()

    def _run_context(
        self, run_name: Optional[str], nested: bool
    ) -> ContextManager[Optional[Any]]:
        """Return context manager for MLflow run (real or dummy).

        Args:
            run_name: Run name or None.
            nested: Whether the run is nested.

        Returns:
            ContextManager yielding mlflow.ActiveRun or None.
        """
        if not self.enabled:

            @contextmanager
            def _dummy() -> Iterator[Optional[Any]]:
                yield None

            return _dummy()

        # Clean up previously active run for safety
        if mlflow.active_run() is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

        return self.run_manager.start_run(run_name, nested=nested)

    @contextmanager
    def start_run(
        self, run_name: Optional[str] = None, nested: bool = False
    ) -> Iterator[Optional[Any]]:
        """Safely start/close an MLflow run.

        Args:
            run_name: Optional descriptive run name.
            nested: Whether nested runs are allowed.

        Yields:
            Active mlflow run object, or None if MLflow disabled.
        """
        active_run: Optional[Any] = None
        try:
            with self._run_context(run_name, nested) as active_run:
                yield active_run
        except Exception as exc:
            # Best-effort exception logging
            if active_run is not None:
                try:
                    mlflow.log_param("error", str(exc))
                    mlflow.set_tag("status", "failed")
                except Exception:
                    pass
            raise

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters (safe no-op if disabled)."""
        if not self.enabled:
            return
        try:
            self.config_logger.log_params(params)
        except Exception as exc:
            print(f"[MLflowManager] Failed to log params: {exc}")

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics (safe no-op if disabled)."""
        if not self.enabled:
            return
        try:
            self.config_logger.log_metrics(metrics, step)
        except Exception as exc:
            print(f"[MLflowManager] Failed to log metrics: {exc}")

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """Log local file as artifact."""
        if not self.enabled:
            return
        if not os.path.exists(local_path):
            print(f"[MLflowManager] Artifact not found: {local_path}")
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as exc:
            print(f"[MLflowManager] Failed to log artifact: {exc}")

    def log_config(self) -> None:
        """Persist configuration into MLflow."""
        if not self.enabled:
            return
        try:
            self.config_logger.log_config(self.cfg)
        except Exception as exc:
            print(f"[MLflowManager] Failed to log config: {exc}")

    def log_model(self, *args: Any, **kwargs: Any) -> None:
        """Log a model using ModelLogger (always returns None)."""
        if not self.enabled:
            return None
        try:
            self.model_logger.log_model(*args, **kwargs)
        except Exception as exc:
            print(f"[MLflowManager] Failed to log model: {exc}")
        return None

    def load_model(self, model_uri: str) -> Any:
        """Load model from MLflow."""
        return self.model_logger.load_model(model_uri)

    def register_model(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        """Register model in MLflow registry."""
        if not self.enabled:
            return None
        try:
            return self.registry_manager.register_model(*args, **kwargs)
        except Exception as exc:
            print(f"[MLflowManager] Failed to register model: {exc}")
            return None
