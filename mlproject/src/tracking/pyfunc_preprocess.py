"""
MLflow PyFunc wrapper for preprocessing logic.

This module provides:
- A `PreprocessPyFunc` class that replays the fitted preprocessing logic
  (fillna, label encoding, scaling) during inference.
- A helper function to log the preprocessing logic as an MLflow PyFunc
  model using the *current running Python environment* (pip freeze).

This design guarantees:
- Environment reproducibility
- No hard-coded dependency versions
- Consistent behavior between training and serving
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel, PythonModelContext
from sklearn.preprocessing import LabelEncoder


class PreprocessPyFunc(PythonModel):
    """
    MLflow PyFunc model that reproduces the exact preprocessing pipeline
    used during training.
    """

    def __init__(self) -> None:
        self.fillna_stats: Dict[str, Dict[str, float]] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = None
        self.scaler_columns: Optional[List[str]] = None

    def load_context(self, context: PythonModelContext) -> None:
        artifacts_dir = context.artifacts["preprocessing_artifacts"]

        with open(os.path.join(artifacts_dir, "fillna_stats.pkl"), "rb") as f:
            self.fillna_stats = pickle.load(f)

        with open(os.path.join(artifacts_dir, "label_encoders.pkl"), "rb") as f:
            self.label_encoders = pickle.load(f)

        with open(os.path.join(artifacts_dir, "scaler.pkl"), "rb") as f:
            scaler_data = pickle.load(f)
            self.scaler = scaler_data.get("scaler")
            self.scaler_columns = scaler_data.get("columns")

    def predict(  # type: ignore[override]
        self,
        context: Any,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Apply preprocessing to input data.
        """
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(
                f"Expected model_input to be pd.DataFrame, got {type(model_input)}"
            )

        df = model_input.copy()

        # 1. Fill missing values
        for col, stats in self.fillna_stats.items():
            if col in df.columns:
                df[col] = df[col].fillna(stats["value"])

        # 2. Label encoding (safe for unseen labels)
        for col, encoder in self.label_encoders.items():
            if col not in df.columns:
                continue

            series = df[col].astype(str)
            known_classes = set(encoder.classes_)
            default_class = encoder.classes_[0]

            safe_series = series.where(series.isin(known_classes), default_class)
            df[col] = encoder.transform(safe_series)

        # 3. Feature scaling
        if self.scaler is not None and self.scaler_columns:
            valid_cols = [c for c in self.scaler_columns if c in df.columns]
            if valid_cols:
                df[valid_cols] = df[valid_cols].astype(np.float64)
                df[valid_cols] = self.scaler.transform(df[valid_cols])

        return df

    def predict_stream(  # pylint: disable=unused-argument
        self,
        context: Any,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Streaming prediction is not supported for this preprocessing PyFunc.

        This method is implemented only to satisfy the abstract interface
        of `mlflow.pyfunc.PythonModel` in MLflow >= 2.x.
        """
        raise NotImplementedError(
            "Streaming prediction is not supported for PreprocessPyFunc."
        )


def _export_current_pip_requirements() -> str:
    """
    Export pip requirements from the current Python environment.

    Returns
    -------
    str
        Path to a temporary requirements.txt file.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as req_file:
        subprocess.check_call(["pip", "freeze"], stdout=req_file)
        return req_file.name


def log_preprocessing_model(
    transform_manager: Any,
    run_id: str,
    artifact_path: str = "preprocessing",
) -> None:
    """
    Log preprocessing logic as an MLflow PyFunc model.

    This function:
    - Collects preprocessing artifacts from TransformManager
    - Captures the *exact* running Python environment (pip freeze)
    - Logs a reproducible MLflow PyFunc model

    Parameters
    ----------
    transform_manager : Any
        Fitted TransformManager instance.
    run_id : str
        Active MLflow run ID.
    artifact_path : str, default="preprocessing"
        Artifact path inside the MLflow run.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = os.path.join(temp_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        # Copy preprocessing artifacts

        src_dir = transform_manager.artifacts_dir
        for filename in (
            "fillna_stats.pkl",
            "label_encoders.pkl",
            "scaler.pkl",
        ):
            src_file = os.path.join(src_dir, filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(artifacts_dir, filename))

        # Capture current environment

        requirements_path = _export_current_pip_requirements()

        # Log PyFunc model

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=PreprocessPyFunc(),
            artifacts={"preprocessing_artifacts": artifacts_dir},
            pip_requirements=requirements_path,
        )

        print(f"Preprocessing model logged at: " f"runs:/{run_id}/{artifact_path}")
