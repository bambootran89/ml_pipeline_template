from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

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

    Returns:
        dict: Processed numeric features ready for model inference.
    """

    engine = PreprocessEngine.instance(cfg)
    return engine.online_transform(df)
