"""
Logging utilities for MLflow: parameters, metrics, and configuration.

Provides ConfigLogger to standardize logging:
- Flattening nested dictionaries for parameter logging
- Logging full Hydra/OmegaConf configuration as YAML
- Logging training/validation metrics
"""

from typing import Any, Dict, List, Optional, Tuple

import mlflow
from omegaconf import OmegaConf


class ConfigLogger:
    """
    Utility class for logging parameters, metrics, and configuration
    to MLflow runs.

    Features:
        - Flatten nested dictionaries for parameter logging
        - Log full OmegaConf/Hydra configuration as YAML
        - Log scalar metrics with optional step
    """

    @staticmethod
    def log_config(cfg: Any) -> None:
        """
        Log the full project configuration as a YAML artifact in MLflow.

        Converts a Hydra/OmegaConf configuration object to a
        standard Python dictionary and logs it under
        "config/full_config.yaml".

        Args:
            cfg (Any): Hydra or OmegaConf configuration object.

        Returns:
            None
        """
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True)
        # Ensure type is dict[str, Any] for mypy
        if isinstance(cfg_dict_raw, dict):
            cfg_dict: Dict[str, Any] = {str(k): v for k, v in cfg_dict_raw.items()}
            mlflow.log_dict(cfg_dict, "config/full_config.yaml")
        else:
            raise TypeError(
                f"Expected OmegaConf container to be dict, got {type(cfg_dict_raw)}"
            )

    @staticmethod
    def flatten_dict(
        d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """
        Recursively flatten a nested dictionary for MLflow logging.

        Args:
            d (Dict[str, Any]): The dictionary to flatten.
            parent_key (str, optional): Prefix for keys during recursion.
                Defaults to "".
            sep (str, optional): Separator between parent and child keys.
                Defaults to ".".

        Returns:
            Dict[str, Any]: A new dictionary with flattened keys.
        """
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ConfigLogger.flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def log_params(params: Dict[str, Any]) -> None:
        """
        Log a dictionary of parameters to MLflow.

        Nested dictionaries are automatically flattened so that
        all keys are compatible with MLflow logging.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to log.

        Returns:
            None
        """
        flat_params = ConfigLogger.flatten_dict(params)
        mlflow.log_params(flat_params)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log a dictionary of metrics to MLflow.

        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values.
            step (Optional[int], optional): Optional step index (e.g., epoch).
                Defaults to None.

        Returns:
            None
        """
        mlflow.log_metrics(metrics, step=step)
