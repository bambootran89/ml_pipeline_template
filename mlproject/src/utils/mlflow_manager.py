"""
MLflow manager để quản lý tracking, artifacts và model registry.
"""
import os
from typing import Any, Dict, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature
from omegaconf import DictConfig, OmegaConf


class MLflowManager:
    """
    Quản lý MLflow tracking, logging và model registry.

    Chức năng:
    - Khởi tạo experiment và run
    - Log parameters, metrics, artifacts
    - Log model với signature
    - Register model vào Model Registry
    - Load model từ registry
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Configuration chứa mlflow settings
        """
        self.cfg = cfg
        self.mlflow_cfg = cfg.get("mlflow", {})

        # Setup tracking URI
        tracking_uri = self.mlflow_cfg.get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)

        # Setup experiment với error handling
        exp_name = self.mlflow_cfg.get("experiment_name", "default")

        try:
            self.experiment = mlflow.set_experiment(exp_name)
        except mlflow.exceptions.MlflowException as e:
            if "deleted experiment" in str(e):
                print(f"[MLflow] Experiment '{exp_name}' was deleted. Restoring...")

                # Try to restore
                client = mlflow.MlflowClient()
                experiments = client.search_experiments(
                    view_type=mlflow.entities.ViewType.DELETED_ONLY
                )

                for exp in experiments:
                    if exp.name == exp_name:
                        client.restore_experiment(exp.experiment_id)
                        print(f"[MLflow] Restored experiment: {exp_name}")
                        self.experiment = mlflow.set_experiment(exp_name)
                        break
                else:
                    # Nếu không restore được, tạo mới với tên khác
                    new_name = f"{exp_name}_v2"
                    print(f"[MLflow] Creating new experiment: {new_name}")
                    self.experiment = mlflow.set_experiment(new_name)
            else:
                raise

        self.run = None
        self.run_id = None

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """
        Bắt đầu MLflow run mới.

        Args:
            run_name: Tên run (optional)

        Returns:
            Active MLflow run
        """
        if run_name is None:
            prefix = self.mlflow_cfg.get("run_name_prefix", "exp")
            run_name = f"{prefix}_{self.experiment.experiment_id}"

        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

        print(f"\n[MLflow] Run started: {run_name}")
        print(f"[MLflow] Run ID: {self.run_id}")
        print(f"[MLflow] Experiment: {self.experiment.name}")

        # Log config
        if self.mlflow_cfg.get("artifacts", {}).get("log_config", True):
            self._log_config()

        return self.run

    def end_run(self):
        """Kết thúc MLflow run."""
        if self.run is not None:
            mlflow.end_run()
            self.run = None
            self.run_id = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters vào MLflow.

        Args:
            params: Dictionary chứa parameters
        """
        # Flatten nested dict
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics vào MLflow.

        Args:
            metrics: Dictionary chứa metrics
            step: Training step (optional)
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model_wrapper: Any,
        artifact_path: str = "model",
        input_example: Optional[np.ndarray] = None,
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
    ):
        """
        Log model vào MLflow với signature.

        Sử dụng PyFunc wrapper để tránh dtype issues.

        Args:
            model_wrapper: Model wrapper instance
            artifact_path: Đường dẫn artifact trong MLflow
            input_example: Ví dụ input để infer signature
            signature: MLflow signature (tự động infer nếu None)
            registered_model_name: Tên model để register (optional)
        """
        if not self.mlflow_cfg.get("artifacts", {}).get("log_model", True):
            return

        # Ensure float32 dtype cho input_example
        if input_example is not None:
            input_example = np.asarray(input_example, dtype=np.float32)

        # Infer signature nếu có input example
        if signature is None and input_example is not None:
            predictions = model_wrapper.predict(input_example)
            signature = infer_signature(input_example, predictions)

        # Wrap model với MLflowModelWrapper
        pyfunc_model = MLflowModelWrapper(model_wrapper)

        # Log as PyFunc model
        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file vào MLflow.

        Args:
            local_path: Đường dẫn file local
            artifact_path: Đường dẫn trong MLflow artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_scaler(self, scaler_path: str):
        """
        Log scaler artifact.

        Args:
            scaler_path: Đường dẫn đến scaler.pkl
        """
        if not self.mlflow_cfg.get("artifacts", {}).get("log_scaler", True):
            return

        if os.path.exists(scaler_path):
            mlflow.log_artifact(scaler_path, "preprocessing")

    def register_model(
        self,
        model_uri: str,
        model_name: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Register model vào Model Registry.

        Args:
            model_uri: URI của model (ví dụ: runs:/<run_id>/model)
            model_name: Tên model trong registry

        Returns:
            ModelVersion object
        """
        if not self.mlflow_cfg.get("registry", {}).get("enabled", True):
            return None

        if model_name is None:
            model_name = self.mlflow_cfg.get("registry", {}).get(
                "model_name", "ts_forecast_model"
            )

        return mlflow.register_model(model_uri, model_name)

    def load_model(self, model_uri: str) -> Any:
        """
        Load model từ MLflow.

        Args:
            model_uri: URI của model

        Returns:
            Loaded model
        """
        return mlflow.pyfunc.load_model(model_uri)

    def _log_config(self):
        """Log toàn bộ config vào MLflow artifacts."""
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        mlflow.log_dict(config_dict, "config.yaml")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.

        Args:
            d: Dictionary cần flatten
            parent_key: Key của parent
            sep: Separator

        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))

        return dict(items)


class MLflowModelWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper để log custom model vào MLflow.
    Wrap model wrapper thành MLflow PyFunc model.

    Fix dtype issues khi serving với MLflow.
    """

    def __init__(self, model_wrapper: Any, preprocessor: Optional[Any] = None):
        """
        Args:
            model_wrapper: Model wrapper instance
            preprocessor: Preprocessor instance (optional)
        """
        self.model_wrapper = model_wrapper
        self.preprocessor = preprocessor

    def predict(self, context, model_input):
        """
        Predict method cho MLflow PyFunc.

        Args:
            context: MLflow context
            model_input: Input data (numpy array or pandas DataFrame)

        Returns:
            Predictions
        """
        import numpy as np

        # Convert to numpy if needed
        if hasattr(model_input, "values"):
            model_input = model_input.values

        # Ensure float32 dtype
        model_input = np.asarray(model_input, dtype=np.float32)

        # Preprocess nếu có
        if self.preprocessor is not None:
            model_input = self.preprocessor.transform(model_input)

        # Predict
        return self.model_wrapper.predict(model_input)
