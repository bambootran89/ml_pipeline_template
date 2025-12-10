"""
Model logging utilities for MLflow.

Provides ModelLogger to handle logging and loading PyFunc models,
including automatic signature inference and optional registry registration.
"""

from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature

from .pyfunc_wrapper import MLflowModelWrapper


class ModelLogger:
    """
    Handles logging and loading MLflow PyFunc models.

    Attributes:
        mlflow_cfg (dict): MLflow-specific configuration block.
        enabled (bool): Flag indicating whether MLflow tracking is enabled.
    """

    def __init__(self, mlflow_cfg: dict, enabled: bool):
        """
        Initialize ModelLogger.

        Args:
            mlflow_cfg (dict): MLflow configuration.
            enabled (bool): Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    def log_model(
        self,
        model_wrapper: Any,
        artifact_path: str = "model",
        input_example: Optional[Any] = None,
        signature: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log a wrapped model as an MLflow PyFunc model.

        Converts the model_wrapper into a PyFunc model, infers the signature
        from the input_example if not provided, and logs the model to MLflow.
        Optionally registers the model in the MLflow Registry.

        Args:
            model_wrapper (Any): Model object implementing predict().
            artifact_path (str, optional):
            Path within the MLflow run to log the model. Defaults to "model".
            input_example (Optional[Any], optional):
              Example input for schema inference. Defaults to None.
            signature (Optional[Any], optional):
            MLflow signature object. Defaults to None.
            registered_model_name (Optional[str], optional):
            Name to register the model in MLflow Registry. Defaults to None.

        Returns:
            None
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

    def load_model(self, model_uri: str) -> mlflow.pyfunc.PyFuncModel:
        """
        Load a PyFunc model from MLflow.

        Args:
            model_uri (str): MLflow model URI (e.g., runs:/<run_id>/model).

        Returns:
            mlflow.pyfunc.PyFuncModel: Loaded PyFunc model.
        """
        print(f"[MLflow] Loading model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
