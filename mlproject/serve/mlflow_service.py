"""
Service load model từ MLflow Model Registry để serving.
"""
import os
from typing import Dict, List

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request

CONFIG_PATH = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")


class MLflowModelService:
    """
    Service quản lý model từ MLflow Registry để serving.

    Chức năng:
    - Load model từ registry (latest hoặc specific version)
    - Preprocess input data
    - Predict với loaded model
    """

    def __init__(
        self,
        model_name: str = "ts_forecast_model",
        model_version: str = "latest",
        cfg_path: str = CONFIG_PATH,
    ):
        """
        Args:
            model_name: Tên model trong MLflow registry
            model_version: Version ("latest", "1", "2", ...)
            cfg_path: Path to config file
        """
        self.model_name = model_name
        self.model_version = model_version
        self.cfg = ConfigLoader.load(cfg_path)

        self.model = None
        self.input_chunk_length = None

    def load_model(self):
        """
        Load model từ MLflow Model Registry.

        Raises:
            RuntimeError: Nếu không load được model
        """
        try:
            # Construct model URI
            if self.model_version == "latest":
                model_uri = f"models:/{self.model_name}/latest"
            else:
                model_uri = f"models:/{self.model_name}/{self.model_version}"

            print(f"[MLflow] Loading model from: {model_uri}")

            # Load model
            self.model = mlflow.pyfunc.load_model(model_uri)

            # Get input_chunk_length từ config
            experiment = self.cfg.experiment
            self.input_chunk_length = experiment.get("hyperparams", {}).get(
                "input_chunk_length", 24
            )

            print(f"[MLflow] Model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow: {e}") from e

    def preprocess(self, data_dict: Dict[str, List]) -> pd.DataFrame:
        """
        Preprocess raw input dictionary.

        Args:
            data_dict: Raw input data

        Returns:
            Preprocessed DataFrame
        """
        df = pd.DataFrame(data_dict)
        if "date" in df.columns:
            df = df.set_index("date")
        return serve_preprocess_request(df, self.cfg)

    def prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract input window từ preprocessed data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Input array [1, seq_len, n_features]

        Raises:
            ValueError: Nếu data không đủ dài
        """
        if len(df) < self.input_chunk_length:
            raise ValueError(
                f"Input data has {len(df)} rows, "
                f"need at least {self.input_chunk_length}"
            )

        # Get last window
        window = df.iloc[-self.input_chunk_length :].values

        # Add batch dimension
        return window[np.newaxis, :]

    def predict(self, data_dict: Dict[str, List]) -> list:
        """
        Predict từ raw input data.

        Args:
            data_dict: Raw input dictionary

        Returns:
            List of predictions

        Raises:
            RuntimeError: Nếu model chưa được load
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess
        df_transformed = self.preprocess(data_dict)

        # Prepare input window
        x_input = self.prepare_input_window(df_transformed)

        # CRITICAL FIX: Ensure float32 dtype trước khi predict
        x_input = np.asarray(x_input, dtype=np.float32)

        # Predict
        preds = self.model.predict(x_input)

        return preds.flatten().tolist()
