import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

from .base import ARTIFACT_DIR
from .engine import PreprocessEngine


def load_scaler(path: str = ""):
    """
    Load a saved scaler and its columns from a pickle file.

    Args:
        path (str, optional): Path to the scaler pickle file.
                             Defaults to 'mlproject/artifacts/preprocessing/scaler.pkl'.

    Returns:
        tuple: (scaler object or None, list of column names or None)
    """
    p = path or os.path.join(ARTIFACT_DIR, "scaler.pkl")
    if not os.path.exists(p):
        return None, None
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj.get("scaler"), obj.get("columns")


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
