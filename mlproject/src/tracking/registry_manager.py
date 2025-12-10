"""
MLflow Model Registry management utilities.
"""

import mlflow


class RegistryManager:
    """
    Handles registering models into MLflow Model Registry.
    """

    def __init__(self, mlflow_cfg, enabled: bool):
        """
        Args:
            mlflow_cfg: MLflow config block.
            enabled: Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    def register_model(self, model_uri: str, model_name=None):
        """
        Register a model into MLflow Model Registry.

        Args:
            model_uri: Path to model in MLflow.
            model_name: Optional registry name.

        Returns:
            Model version or None.
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
