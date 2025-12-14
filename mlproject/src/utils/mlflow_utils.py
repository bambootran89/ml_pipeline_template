from typing import Any, Optional

import mlflow
from omegaconf import DictConfig


def load_model_from_registry_safe(
    cfg: DictConfig,
    default_model_name: str,
) -> Optional[Any]:
    """
    Load a prediction model from MLflow Model Registry.

    Parameters
    ----------
    cfg : DictConfig
        Full application configuration.
    default_model_name : str
        Fallback model name if not specified in config.

    Returns
    -------
    Optional[Any]
        Loaded MLflow model, or None if loading fails.
    """
    try:
        registry_cfg = cfg.get("mlflow", {}).get("registry", {})
        model_name: str = registry_cfg.get("model_name", default_model_name)
        model_uri = f"models:/{model_name}/latest"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        return None


def load_companion_preprocessor_from_model(
    model: Any,
    artifact_path: str = "preprocessing_pipeline",
) -> Optional[Any]:
    """
    Load the preprocessing PyFunc model associated with a prediction model.

    The preprocessing model is resolved via the run_id stored in
    MLflow model metadata.

    Parameters
    ----------
    model : Any
        Loaded MLflow prediction model.
    artifact_path : str, default="preprocessing_pipeline"
        Artifact path used when logging the preprocessing model.

    Returns
    -------
    Optional[Any]
        Loaded preprocessing PyFunc model, or None if unavailable.
    """
    metadata = getattr(model, "metadata", None)
    run_id: Optional[str] = getattr(metadata, "run_id", None)

    if not run_id:
        return None

    preprocessor_uri = f"runs:/{run_id}/{artifact_path}"

    try:
        return mlflow.pyfunc.load_model(preprocessor_uri)
    except Exception:
        return None
