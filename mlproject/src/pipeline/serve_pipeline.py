import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request
from mlproject.src.utils.mlflow_manager import MLflowManager


class TestPipeline(BasePipeline):
    """
    Inference / online serving pipeline.

    Simple flow:
    1. Preprocess raw input → transformed features
    2. Model predict → output

    NO DataModule, NO train/val/test split, NO batching.
    """

    def __init__(self, cfg_path=""):
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        # Initialize MLflow Manager
        self.mlflow = (
            MLflowManager(self.cfg)
            if self.cfg.get("mlflow", {}).get("enabled", False)
            else None
        )

    def preprocess(self, data=None):
        """
        Transform raw data using saved scaler.

        Args:
            data: Raw DataFrame with historical data.

        Returns:
            pd.DataFrame: Transformed features ready for inference.
        """
        if data is None:
            raise ValueError("Test mode requires input data")
        return serve_preprocess_request(data, self.cfg)

    def _load_model(self, approach):
        """
        Load trained model.
        Priority: MLflow Registry > Local Artifacts.
        """
        # 1. Try Loading from MLflow Registry
        if self.mlflow.enabled:
            try:
                # Determine Model Name from Config (Registry) or Approach
                registry_conf = self.cfg.get("mlflow", {}).get("registry", {})
                model_name = registry_conf.get("model_name", approach["model"])
                version = "latest"  # Or specify version via config/args

                model_uri = f"models:/{model_name}/{version}"
                print(f"[TestPipeline] Loading model from MLflow Registry: {model_uri}")

                # Load as PyFunc model (Generic wrapper)
                return mlflow.pyfunc.load_model(model_uri)

            except Exception as e:
                print(
                    f"[TestPipeline] Warning: Could not load from MLflow ({e}). Falling back to local artifacts."
                )

        # 2. Fallback: Load from Local Artifacts
        print("[TestPipeline] Loading model from Local Artifacts...")
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """
        Extract last window from transformed data for inference.

        Args:
            df: Transformed DataFrame (preprocessed).

        Returns:
            np.ndarray: Input window [1, seq_len, n_features].
        """
        seq_len = input_chunk_length

        # Get last seq_len rows
        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")

        window = df.iloc[-seq_len:].values

        # Add batch dimension [1, seq_len, n_features]
        return window[np.newaxis, :].astype(np.float32)

    def run_approach(self, approach, data):
        """
        Run inference: preprocess → predict.

        Args:
            approach: Experiment approach config.
            data: Raw DataFrame (NOT preprocessed).

        Returns:
            np.ndarray: Model predictions.
        """
        # Step 1: Preprocess
        df_transformed = self.preprocess(data)
        input_chunk_length = approach.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )
        # Step 2: Prepare input window
        x_input = self._prepare_input_window(df_transformed, input_chunk_length)
        # Step 3: Load model and predict
        wrapper = self._load_model(approach)
        preds = wrapper.predict(x_input)

        print(f"[INFERENCE] Input shape: {x_input.shape}, Output shape: {preds.shape}")
        print(f"[INFERENCE] Predictions: {preds.flatten()[:10]}...")  # Show first 10

        return preds
