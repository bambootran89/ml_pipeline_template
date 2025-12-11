from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
from mlflow.models import infer_signature
from mlflow.pyfunc import PyFuncModel

from .pyfunc_wrapper import MLflowModelWrapper


class ModelLogger:
    """
    Utility class to handle logging and loading of MLflow PyFunc models.

    Responsibilities:
    - Convert model wrappers to PyFunc models
    - Infer model signature automatically from example input
    - Log models to MLflow with optional registry registration
    - Load PyFunc models from MLflow

    Attributes:
        mlflow_cfg (dict): MLflow-specific configuration block.
        enabled (bool): Flag indicating whether MLflow tracking is enabled.
    """

    def __init__(self, mlflow_cfg: dict, enabled: bool):
        """
        Initialize ModelLogger.

        Args:
            mlflow_cfg (dict): MLflow configuration block.
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
        Log a model as an MLflow PyFunc model.

        Converts the provided model_wrapper into a PyFunc model,
        infers the signature if not provided, and logs to MLflow.
        Optionally registers the model in MLflow Registry.

        Args:
            model_wrapper (Any): Model implementing predict().
            artifact_path (str, optional):
                Path in MLflow run to save the model. Default is "model".
            input_example (Optional[Any], optional):
                Example input for signature inference. Default is None.
            signature (Optional[Any], optional):
                MLflow signature object. Default is None.
            registered_model_name (Optional[str], optional):
                Name for registry registration. Default is None.

        Returns:
            None
        """
        if not self.enabled:
            return

        if not self.mlflow_cfg.get("artifacts", {}).get("log_model", True):
            return

        # Ensure input_example is numpy array
        if input_example is not None and not hasattr(input_example, "values"):
            input_example = np.asarray(input_example, dtype=np.float32)

        # Infer signature if needed
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

    def load_model(self, model_uri: str) -> PyFuncModel:
        """
        Load a PyFunc model from MLflow.

        Args:
            model_uri (str): MLflow model URI (e.g., runs:/<run_id>/model).

        Returns:
            PyFuncModel: Loaded MLflow PyFunc model.
        """
        print(f"[MLflow] Loading model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)
