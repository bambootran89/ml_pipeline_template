"""
MLflow manager module for tracking, artifacts, and model registry management.
Optimized for MLOps best practices: Autologging, Context Management, and Reproducibility.
"""
import contextlib
import os
from typing import Any, Dict, Generator, Optional

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf


class MLflowManager:
    """
    Manages MLflow tracking, logging, and model registry.

    Advanced features:
    - Context Manager for safe Run management.
    - Autologging for popular frameworks.
    - Automatic Git commit hash logging.
    - Automatic Model Signature and Input Example handling.
    - Model Registry Support.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize MLflowManager.

        Args:
            cfg: Configuration dictionary containing mlflow settings.
        """
        self.cfg = cfg
        self.mlflow_cfg = cfg.get("mlflow", {})
        self.enabled = self.mlflow_cfg.get("enabled", False)

        if not self.enabled:
            return

        # 1. Setup Tracking URI
        tracking_uri = self.mlflow_cfg.get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # 2. Setup Experiment (Robust creation/restoration)
        self._setup_experiment()

        # 3. Setup Autologging
        self._setup_autologging()

        self.run = None
        self.run_id = None

    def _setup_experiment(self):
        """Sets up the experiment, automatically restoring it if deleted."""
        exp_name = self.mlflow_cfg.get("experiment_name", "Default_Experiment")
        try:
            self.experiment = mlflow.set_experiment(exp_name)
        except mlflow.exceptions.MlflowException as e:
            if "deleted experiment" in str(e).lower():
                print(
                    f"[MLflow] Experiment '{exp_name}' was deleted. Attempting to restore..."
                )
                client = mlflow.MlflowClient()
                experiments = client.search_experiments(
                    view_type=mlflow.entities.ViewType.DELETED_ONLY
                )
                for exp in experiments:
                    if exp.name == exp_name:
                        client.restore_experiment(exp.experiment_id)
                        print(f"[MLflow] Restored experiment: {exp_name}")
                        self.experiment = mlflow.set_experiment(exp_name)
                        return

                # Fallback: Create new name if restoration fails
                new_name = f"{exp_name}_v2"
                print(f"[MLflow] Cannot restore. Creating new experiment: {new_name}")
                self.experiment = mlflow.set_experiment(new_name)
            else:
                raise e

    def _setup_autologging(self):
        """Enables autologging for supported libraries."""
        if self.mlflow_cfg.get("autolog", True):
            # General autolog (covers sklearn, xgboost, statsmodels, etc.)
            # log_models=False so we can control custom pyfunc model wrapper saving
            mlflow.autolog(log_models=False, exclusive=False)
            print("[MLflow] Autologging enabled.")

    @contextlib.contextmanager
    def start_run(
        self, run_name: Optional[str] = None
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """
        Context Manager to start an MLflow run.
        """
        if not self.enabled:
            yield None
            return

        # Create default run name if missing
        if run_name is None:
            prefix = self.mlflow_cfg.get("run_name_prefix", "run")
            run_name = f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        # Start Run
        with mlflow.start_run(run_name=run_name) as run:
            self.run = run
            self.run_id = run.info.run_id
            print(f"\n[MLflow] Run started: {run_name} (ID: {self.run_id})")

            # Log basic environment info
            self._log_environment_info()

            # Log Config
            if self.mlflow_cfg.get("artifacts", {}).get("log_config", True):
                self._log_config()

            try:
                yield run
            finally:
                self.run = None
                self.run_id = None

    def _log_environment_info(self):
        """Logs Git commit hash and system info for Reproducibility."""
        # Log Git Commit
        try:
            import git

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            mlflow.set_tag("git_commit", sha)
            if repo.is_dirty():
                mlflow.set_tag("git_dirty", "True")
        except (ImportError, Exception):
            # Suppress warning if git is not installed or not a git repo
            pass

        # Log User
        import getpass

        mlflow.set_tag("user", getpass.getuser())

    def log_params(self, params: Dict[str, Any]):
        """Logs parameters (Flattened)."""
        if not self.enabled:
            return
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Logs metrics."""
        if not self.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model_wrapper: Any,
        artifact_path: str = "model",
        input_example: Optional[Any] = None,
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
    ):
        """
        Logs custom model wrapper as a PyFunc model with Signature.
        """
        if not self.enabled:
            return
        if not self.mlflow_cfg.get("artifacts", {}).get("log_model", True):
            return

        # Handle Input Example
        if input_example is not None and hasattr(input_example, "values"):
            pass
        elif input_example is not None:
            input_example = np.asarray(input_example, dtype=np.float32)

        # Auto-infer Signature if missing
        if signature is None and input_example is not None:
            try:
                preds = model_wrapper.predict(input_example)
                signature = infer_signature(input_example, preds)
            except Exception as e:
                print(f"[MLflow] Warning: Could not infer signature. Error: {e}")

        # Wrap model
        pyfunc_model = MLflowModelWrapper(model_wrapper)

        print(f"[MLflow] Logging model to artifact path: '{artifact_path}'")
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,  # Support direct registration
        )

    def register_model(
        self,
        model_uri: str,
        model_name: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Register a model to the MLflow Model Registry.

        Args:
            model_uri: The URI of the model (e.g., 'runs:/<run_id>/model').
            model_name: The name in the registry. If None, uses config.

        Returns:
            ModelVersion: The registered model version object.
        """
        if not self.enabled:
            return None
        if not self.mlflow_cfg.get("registry", {}).get("enabled", True):
            return None

        if model_name is None:
            model_name = self.mlflow_cfg.get("registry", {}).get(
                "model_name", "ts_forecast_model"
            )

        print(f"[MLflow] Registering model URI '{model_uri}' to name '{model_name}'")
        return mlflow.register_model(model_uri, model_name)

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow (Registry or Run).

        Args:
            model_uri: The URI of the model (e.g., 'models:/MyModel/Production').

        Returns:
            The loaded PyFunc model (wrapper).
        """
        print(f"[MLflow] Loading model from {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs any file."""
        if self.enabled and os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
        elif self.enabled:
            print(f"[MLflow] Warning: Artifact not found at {local_path}")

    def _log_config(self):
        """Logs full config as a YAML file."""
        if not self.enabled:
            return
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow.log_dict(config_dict, "config/full_config.yaml")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flattens nested dictionary for params logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper class to standardize Serving via MLflow.
    Ensures input data type (float32) before passing to the original model.
    """

    def __init__(self, model_wrapper: Any):
        self.model_wrapper = model_wrapper

    def predict(self, context, model_input):
        """
        Predict method called when serving model or load_model().predict().
        """
        # 1. Standardize Input to Numpy Array Float32
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values

        # Ensure safe data type for model (especially Torch/TF)
        model_input = np.asarray(model_input, dtype=np.float32)

        # 2. Call original model's predict
        return self.model_wrapper.predict(model_input)
