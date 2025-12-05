import os
import pickle
from typing import Any, Dict

import pandas as pd

ARTIFACT_DIR = os.path.join("mlproject", "artifacts", "preprocessing")


def load_scaler(path: str = ""):
    """
    Load a saved scaler and its columns from a pickle file.

    Args:
        path (str, optional): Path to the scaler pickle file. Defaults to
                              'mlproject/artifacts/preprocessing/scaler.pkl'.

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
    features: Dict[str, Any], scaler=None, scaler_columns=None
) -> Dict[str, float]:
    """
    Preprocess a single request of raw features online (fill missing and scale).

    Args:
        features (dict): Raw features for a single request.
        scaler (optional): Fitted sklearn scaler object.
        scaler_columns (optional): List of column names expected by the scaler.

    Returns:
        dict: Processed numeric features.
    """
    df = pd.DataFrame([features])
    # ensure columns exist
    if scaler is not None and scaler_columns is not None:
        for c in scaler_columns:
            if c not in df.columns:
                df[c] = 0.0
        df[scaler_columns] = scaler.transform(df[scaler_columns].values)
    # fillna simple
    df = df.fillna(0)
    return df.iloc[0].to_dict()
