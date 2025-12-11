"""
Utility functions for handling time-series input/output shapes.

These helpers standardize tensor/array dimensionality for traditional
machine-learning models and provide automatic inference of model
input/output dimensions.
"""

from typing import Any, Tuple

import numpy as np


def flatten_timeseries(x: np.ndarray) -> np.ndarray:
    """
    Flatten a 3D time-series array into 2D format.

    This is useful for traditional machine-learning models
    (e.g., XGBoost, RandomForest, Linear models) which expect
    2D input matrices.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (batch, seq_len, features) or any numeric array.

    Returns
    -------
    np.ndarray
        If x is 3D, returns array of shape (batch, seq_len * features).
        Otherwise, returns x unchanged.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3:
        batch, seq_len, n_feat = x.shape
        return x.reshape(batch, seq_len * n_feat)

    return x


def infer_io_dims(x: Any, y: Any) -> Tuple[int, int]:
    """
    Infer the input and output dimensionality from sample batches.

    Supports NumPy arrays, PyTorch tensors, and most objects
    exposing a `.shape` attribute.

    Rules
    -----
    - For 3D inputs: (batch, seq_len, features) → input_dim = seq_len * features
    - For 2D inputs: (batch, features) → input_dim = features
    - For 1D or unknown structures → input_dim = last dimension or 1
    - For y:
        - 2D: (batch, outputs) → output_dim = outputs
        - 1D → output_dim = 1
        - non-array → output_dim = 1

    Parameters
    ----------
    x : Any
        Input batch, typically numpy array or tensor.
    y : Any
        Target batch.

    Returns
    -------
    Tuple[int, int]
        A tuple (input_dim, output_dim).
    """
    # -------- Infer input_dim --------
    if hasattr(x, "shape"):
        ndim = len(x.shape)

        if ndim == 3:  # (batch, seq, feat)
            _, seq_len, feat = x.shape
            input_dim = seq_len * feat
        elif ndim == 2:  # (batch, feat)
            input_dim = x.shape[1]
        else:  # fallback (batch, ?)
            input_dim = x.shape[-1] if x.shape else 1
    else:
        input_dim = 1

    # -------- Infer output_dim --------
    if hasattr(y, "shape"):
        if len(y.shape) > 1:
            output_dim = y.shape[1]
        else:
            output_dim = 1
    else:
        output_dim = 1

    return input_dim, output_dim

def to_list_if_array(arr: Any) -> list:
    """Converts numpy array to list, handles other iterables."""
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)
