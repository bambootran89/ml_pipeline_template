from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .engine import PreprocessEngine


def online_preprocess_request(
    features: Dict[str, Any],
) -> Dict[str, float]:
    """
    Preprocess a single request of raw features online (fill missing and scale).

    Args:
        features (dict): Raw features for a single request.

    Returns:
        dict: Processed numeric features ready for model inference.
    """

    df = pd.DataFrame([features])

    engine = PreprocessEngine.instance()
    df = engine.online_transform(df)

    return df.iloc[0].to_dict()


def serve_preprocess_request(
    df: pd.DataFrame, cfg: Optional[DictConfig] = None
) -> pd.DataFrame:
    """
    Preprocess test df (fill missing and scale).

    Args:
        df (DataFrame): DataFrame
        cfg (DictConfig, optional): Preprocessing configuration.

    Returns:
        pd.DataFrame: Processed numeric features ready for model inference.
    """
    # Fix mypy error: Argument 1 to "instance" has incompatible type "DictConfig | None"
    cfg_dict: Optional[Dict[Any, Any]] = None
    if cfg is not None:
        # Convert DictConfig to standard python dict
        container = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(container, dict):
            cfg_dict = container
        else:
            # Fallback for unlikely case where cfg is a ListConfig
            cfg_dict = {}

    engine = PreprocessEngine.instance(cfg_dict)
    return engine.online_transform(df)
