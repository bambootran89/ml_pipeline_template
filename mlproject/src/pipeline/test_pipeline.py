import numpy as np
import pandas as pd

from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import test_preprocess_request


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
        return test_preprocess_request(data, self.cfg)

    def _load_model(self, approach):
        """
        Load trained model from artifacts.

        Args:
            approach: Experiment approach config.

        Returns:
            Model wrapper with loaded weights.
        """
        name = approach["model"]
        hp = approach.get("hyperparams", {})

        if name == "nlinear":
            wrapper = NLinearWrapper(hp)
        elif name == "tft":
            wrapper = TFTWrapper(hp)
        else:
            raise RuntimeError(f"Unknown model {name}")

        wrapper.load(self.cfg.training.artifacts_dir)
        return wrapper

    def _prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract last window from transformed data for inference.

        Args:
            df: Transformed DataFrame (preprocessed).

        Returns:
            np.ndarray: Input window [1, seq_len, n_features].
        """
        seq_len = self.cfg.data.get("seq_len", 96)

        # Get last seq_len rows
        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")

        window = df.iloc[-seq_len:].values

        # Add batch dimension [1, seq_len, n_features]
        return window[np.newaxis, :]

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

        # Step 2: Prepare input window
        x_input = self._prepare_input_window(df_transformed)

        # Step 3: Load model and predict
        wrapper = self._load_model(approach)
        preds = wrapper.predict(x_input)

        print(f"[INFERENCE] Input shape: {x_input.shape}, Output shape: {preds.shape}")
        print(f"[INFERENCE] Predictions: {preds.flatten()[:10]}...")  # Show first 10

        return preds
