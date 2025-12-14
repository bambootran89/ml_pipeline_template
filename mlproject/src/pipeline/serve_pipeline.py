"""
TestPipeline: Simple inference pipeline for online serving.

Responsibilities:
- Load trained models from MLflow Registry or local artifacts.
- Preprocess raw input data for serving.
- Build input windows for time series models.
- Run forward prediction and return outputs.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.online import OnlinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.utils.mlflow_utils import (
    load_companion_preprocessor_from_model,
    load_model_from_registry_safe,
)


class TestPipeline(BasePipeline):
    """
    Inference pipeline for online evaluation.

    This pipeline does not use DataModule, dataset splits, or batching logic.
    """

    def __init__(self, cfg_path: str = "") -> None:
        """
        Initialize pipeline and load configuration.

        Args:
            cfg_path: Path to configuration file. If empty, default rules apply.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)

        self.mlflow_manager = MLflowManager(self.cfg)
        self.local_preprocessor = OnlinePreprocessor(self.cfg)

        self.model: Any = None
        self.preprocessor_model: Any | None = None

        self.model = self._load_model_from_mlflow(self.cfg.get("experiment"))

    def _load_model_from_mlflow(self, approach_cfg: DictConfig) -> Any | None:
        """
        Load the prediction model from MLflow Model Registry for inference and
        resolve its companion preprocessing model if available.

        The method attempts to:
        1. Load the latest version of the prediction model specified by
        ``approach_cfg.model`` from the MLflow Model Registry.
        2. Load the associated preprocessing PyFunc model using the ``run_id``
        stored in the prediction model metadata.
        3. Gracefully fall back to local preprocessing logic if the companion
        preprocessing model cannot be loaded.

        Parameters
        ----------
        approach_cfg : DictConfig
            Model configuration containing the model name and hyperparameters.

        Returns
        -------
        Any | None
            Loaded MLflow prediction model, or ``None`` if the model cannot be
            loaded from the MLflow Model Registry.
        """
        model = load_model_from_registry_safe(
            cfg=self.cfg,
            default_model_name=approach_cfg.model,
        )

        if model is None:
            print("[TestPipeline] MLflow model load failed")
            return None

        self.preprocessor_model = load_companion_preprocessor_from_model(model)

        if self.preprocessor_model is None:
            print(
                "[TestPipeline] Companion preprocessor not available. "
                "Using local fallback."
            )

        return model

    def preprocess(self, data: Any = None) -> Any:
        """
        Preprocess input data for serving.

        Args:
            data: Optional raw input DataFrame.

        Returns:
            Preprocessed DataFrame or None if no data provided.
        """
        if self.preprocessor_model is not None:
            return self.preprocessor_model.predict(data)

        return self.local_preprocessor.transform(data)

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """
        Build model input window for prediction.

        Args:
            df: Preprocessed DataFrame.
            input_chunk_length: Sequence length required by model.

        Returns:
            Model input array with shape [1, seq_len, n_features].

        Raises:
            ValueError: If input has fewer rows than input_chunk_length.
        """
        seq_len: int = input_chunk_length
        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")
        window: np.ndarray = df.iloc[-seq_len:].values
        return window[np.newaxis, :].astype(np.float32)

    def run_approach(self, approach: Dict[str, Any], data: pd.DataFrame) -> np.ndarray:
        """
        Execute model inference for a given approach.

        Steps:
        1. Preprocess raw data.
        2. Build input window.
        3. Load model from MLflow or local artifacts.
        4. Run forward prediction.

        Args:
            approach: Dictionary containing model name and hyperparameters.
            data: Raw historical DataFrame.

        Returns:
            Numpy array of predictions.
        """
        if isinstance(approach, DictConfig):
            approach = OmegaConf.to_container(approach, resolve=True)

        if not isinstance(approach, dict):
            raise TypeError(f"Expected approach to be dict, got {type(approach)}")

        df_transformed: pd.DataFrame = self.preprocess(data)
        input_chunk_length: int = approach.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )
        x_input: np.ndarray = self._prepare_input_window(
            df_transformed, input_chunk_length
        )

        preds: np.ndarray = self.model.predict(x_input)

        print(f"[INFERENCE] Input shape: {x_input.shape}, Output shape: {preds.shape}")
        print(f"[INFERENCE] Predictions (first 10): {preds.flatten()[:10]}...")

        return preds
