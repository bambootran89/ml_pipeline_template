"""
Utility functions for handling time-series input/output shapes.

These helpers standardize tensor/array dimensionality for traditional
machine-learning models and provide automatic inference of model
input/output dimensions.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


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


def flatten_metrics_for_mlflow(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Flatten a metrics dictionary into an MLflow-compatible format.

    MLflow only accepts scalar float values for logging. However, many
    time-series metrics may return vector outputs (e.g., multivariate or
    multi-horizon forecasts). This utility converts any `np.ndarray`
    metric values into multiple scalar metrics using the pattern:

        <metric_name>_<index> = float_value

    Example:
        Input:
            {
                "mae": np.array([0.12, 0.20, 0.31]),
                "rmse": 0.55,
            }

        Output:
            {
                "mae_0": 0.12,
                "mae_1": 0.20,
                "mae_2": 0.31,
                "rmse": 0.55,
            }

    Args:
        metrics (dict[str, Any]):
            Dictionary of metrics returned by the evaluator. Values may be
            scalars, numpy arrays, or unsupported types (ignored).

    Returns:
        dict[str, float]:
            A flattened metrics dictionary containing only scalar floats,
            safe for logging with MLflow.

    Notes:
        - Unsupported types are silently skipped.
        - Vector metrics preserve ordering (index corresponds to output dimension).
    """
    result: dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                result[f"{key}_{i}"] = float(v)
        elif isinstance(val, (np.floating, float, int)):
            result[key] = float(val)
        else:
            # Skip unsupported types
            continue
    return result


def load_data_csv(
    path: str,
    index_col: Optional[str] = None,
    data_type: str = "timeseries",
) -> pd.DataFrame:
    """
    Load a CSV dataset with robust handling for both time-series and tabular data.

    This method supports:
    - Time-series datasets with a datetime index column
    - Tabular datasets without a dedicated index column

    Parameters
    ----------
    path : str
        Path to the CSV file.
    index_col : Optional[str], default=None
        Name of the column to be used as index. Required for time-series data.
    data_type : str, default="timeseries"
        Dataset type. Must be either "timeseries" or "tabular".

    Returns
    -------
    pd.DataFrame
        Loaded dataset with appropriate index configuration.

    Raises
    ------
    ValueError
        If `data_type` is invalid or if `index_col` is missing for time-series data.
    """
    if data_type not in {"timeseries", "tabular"}:
        raise ValueError(f"Unsupported data_type: {data_type}")

    try:
        columns = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError:
        logger.warning("CSV file is empty: %s", path)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Time-series dataset
    # ------------------------------------------------------------------
    if data_type == "timeseries":
        if not index_col:
            raise ValueError("index_col must be provided for time-series data")

        if index_col not in columns:
            raise ValueError(
                f"Index column '{index_col}' \
                    not found in CSV columns: {list(columns)}"
            )

        try:
            df = pd.read_csv(path, parse_dates=[index_col])
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse '%s' as datetime. Falling back to raw index.",
                index_col,
            )
            df = pd.read_csv(path)

        return df.set_index(index_col)

    # ------------------------------------------------------------------
    # Tabular dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(path)

    if index_col and index_col in df.columns:
        logger.info("Using column '%s' as index for tabular dataset.", index_col)
        df = df.set_index(index_col)

    return df


def select_columns(
    cfg: Any,
    df: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Select a subset of columns from the DataFrame based on configuration.

    The selection logic is:
    - Feature columns are taken from ``cfg.data.feature_cols``.
    - Target columns are taken from ``cfg.data.target_columns`` if
    ``include_target`` is True.
    - Only columns that actually exist in the DataFrame are kept.
    - If no valid columns can be resolved, the original DataFrame is returned
    for backward compatibility and easier debugging.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    include_target : bool, default=True
        Whether to include target columns in the output.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only selected columns, or the original
        DataFrame if no valid columns are found.
    """
    data_cfg = cfg.get("data", {})

    feature_cols: Sequence[str] = data_cfg.get("feature_cols") or []

    # Backward compatibility: if feature_cols is not defined, keep original DF
    if not feature_cols:
        return df

    cols_to_keep: List[str] = list(feature_cols)

    if include_target:
        target_cols = data_cfg.get("target_columns", [])

        if isinstance(target_cols, str):
            cols_to_keep.append(target_cols)
        elif isinstance(target_cols, (list, tuple)):
            cols_to_keep.extend(target_cols)

    # Keep only columns that actually exist in the DataFrame
    valid_cols: List[str] = [col for col in cols_to_keep if col in df.columns]

    if not valid_cols:
        # Strategy: return original DF to avoid hard failure and aid debugging
        return df

    return df.loc[:, valid_cols]


def load_model_from_registry(model_name: str, version: str = "latest"):
    """
    Load the MLflow PyFunc model from the model registry.
    Returns the loaded model object.
    """
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e


def sync_artifacts_from_registry(model_name, artifacts_dir) -> None:
    """
    Synchronize preprocessing artifacts from the latest registered model version.

    This method downloads all preprocessing-related artifacts (e.g. fillna
    statistics, label encoders, scalers) associated with the latest version
    of the registered model and stores them locally for inference or reuse.

    Expected artifact structure in MLflow run:
        preprocessing/
            fillna_stats.pkl
            label_encoders.pkl
            scaler.pkl
            ...

    Missing artifacts are tolerated and will not raise errors.
    """
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(model_name, stages=None)
        if not versions:
            print(f"[MLflow] No registered model found for '{model_name}'")
            return

        latest_version = max(versions, key=lambda v: int(v.version))
        run_id = latest_version.run_id

        local_artifacts_dir = artifacts_dir
        os.makedirs(local_artifacts_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download entire preprocessing artifact folder
            client.download_artifacts(run_id, "preprocessing", tmp_dir)

            src_root = os.path.join(tmp_dir, "preprocessing")
            if not os.path.isdir(src_root):
                print("[MLflow] No preprocessing artifacts found in run.")
                return

            __copy_artifacts(
                src_dir=src_root,
                dst_dir=local_artifacts_dir,
                allowed_ext={".pkl", ".json", ".yaml"},
            )

    except Exception as exc:  # pylint: disable=broad-except
        print(f"[MLflow] Error syncing preprocessing artifacts: {exc}")


def __copy_artifacts(
    src_dir: str,
    dst_dir: str,
    allowed_ext: Iterable[str],
) -> None:
    """
    Copy artifact files from source directory to destination directory.

    Parameters
    ----------
    src_dir : str
        Source directory containing downloaded artifacts.
    dst_dir : str
        Destination directory for local storage.
    allowed_ext : Iterable[str]
        Allowed file extensions to copy.
    """
    for root, _, files in os.walk(src_dir):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext not in allowed_ext:
                continue

            src_path = os.path.join(root, filename)
            dst_path = os.path.join(dst_dir, filename)

            shutil.copy2(src_path, dst_path)
