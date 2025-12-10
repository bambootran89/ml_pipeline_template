"""
MLflow Model Registry utilities.

Provides RegistryManager to handle registration of models into
MLflow Model Registry.
"""

from typing import Optional

import mlflow


class RegistryManager:
    """
    Handles registering models into MLflow Model Registry.

    Attributes:
        mlflow_cfg (dict): MLflow-specific configuration.
        enabled (bool): Flag indicating whether MLflow tracking is enabled.
    """

    def __init__(self, mlflow_cfg: dict, enabled: bool):
        """
        Initialize RegistryManager.

        Args:
            mlflow_cfg (dict): MLflow configuration dictionary.
            enabled (bool): Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    def register_model(
        self, model_uri: str, model_name: Optional[str] = None
    ) -> Optional[mlflow.entities.ModelVersion]:
        """
        Register a model in the MLflow Model Registry.

        If registry is disabled in configuration or MLflow tracking
        is not enabled, returns None. If no model_name is provided,
        uses the default from configuration or 'ts_model'.

        Args:
            model_uri (str): MLflow model URI (e.g., runs:/<run_id>/model).
            model_name (Optional[str], optional): Name to register the model.
                Defaults to None.

        Returns:
            Optional[mlflow.entities.ModelVersion]:
              Registered model version, or None if disabled.
        """
        if not self.enabled:
            return None

        if not self.mlflow_cfg.get("registry", {}).get("enabled", True):
            return None

        if model_name is None:
            model_name = self.mlflow_cfg.get("registry", {}).get(
                "model_name", "ts_model"
            )

        print(f"[MLflow] Registering model '{model_uri}' as '{model_name}'")
        return mlflow.register_model(model_uri, model_name)
