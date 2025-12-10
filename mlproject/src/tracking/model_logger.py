"""
Model logging utilities for MLflow:
- log_model() for pyfunc custom models
- load_model() helper
"""

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature

from .pyfunc_wrapper import MLflowModelWrapper


class ModelLogger:
    """
    Handles logging pyfunc models and loading models back.
    """

    def __init__(self, mlflow_cfg, enabled: bool):
        """
        Args:
            mlflow_cfg: MLflow config.
            enabled: Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    def log_model(
        self,
        model_wrapper,
        artifact_path="model",
        input_example=None,
        signature=None,
        registered_model_name=None,
    ):
        """
        Log wrapped model as MLflow pyfunc.

        Args:
            model_wrapper: Custom wrapper implementing predict().
            artifact_path: MLflow artifact folder path.
            input_example: Optional example input.
            signature: Optional MLflow signature.
            registered_model_name: Optional registry name.
        """
        if not self.enabled:
            return

        if not self.mlflow_cfg.get("artifacts", {}).get("log_model", True):
            return

        if input_example is not None and not hasattr(input_example, "values"):
            input_example = np.asarray(input_example, dtype=np.float32)

        if signature is None and input_example is not None:
            try:
                preds = model_wrapper.predict(input_example)
                signature = infer_signature(input_example, preds)
            except Exception as e:
                print(f"[MLflow] Warning: signature inference failed ({e})")

        pyfunc_model = MLflowModelWrapper(model_wrapper)

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

    def load_model(self, model_uri: str):
        """
        Load a pyfunc model from MLflow.

        Args:
            model_uri: e.g. "runs:/<id>/model"

        Returns:
            PythonModel
        """
        print(f"[MLflow] Loading model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
