"""
TestPipeline module.

Provides a simple inference pipeline for online serving.
Does not use DataModule, dataset splits, or batching logic.
"""

from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request
from mlproject.src.tracking.mlflow_manager import MLflowManager


class TestPipeline(BasePipeline):
    """
    Inference pipeline used for online evaluation.

    Responsibilities:
    - Load trained models from MLflow or local storage.
    - Preprocess raw input for serving.
    - Build input windows for time series models.
    - Run forward prediction and return model output.
    """

    def __init__(self, cfg_path: str = "") -> None:
        """
        Initialize pipeline and load configuration.

        Creates MLflowManager when MLflow is enabled in
        configuration. Also initializes BasePipeline.

        Args:
            cfg_path: Path to configuration file. If empty,
                default loader rules apply.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.mlflow_manager = MLflowManager(self.cfg)

    def preprocess(self, data: Any = None) -> Any:
        """
        Override abstract BasePipeline.preprocess.

        Accepts optional data for serving-time preprocessing.
        """
        if data is None:
            return None
        return self._preprocess_input(data)

    def _preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply serving-time preprocessing to raw input.

        Args:
            data: Raw DataFrame containing historical input.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for model inference.

        Raises:
            ValueError: When no input data is provided.
        """
        if data is None:
            raise ValueError("Test mode requires input data")
        return serve_preprocess_request(data, self.cfg)

    def _load_model(self, approach: Dict[str, Any]) -> Any:
        """
        Load a trained model based on configuration.

        Priority:
        1. MLflow Registry (when enabled).
        2. Local artifacts directory (fallback).

        Args:
            approach: Dict specifying model name and hyperparameters.

        Returns:
            Loaded model object (PyFunc or custom wrapper).
        """
        approach_cfg: DictConfig = DictConfig(approach)

        if self.mlflow_manager and self.mlflow_manager.enabled:
            try:
                registry_conf = self.cfg.get("mlflow", {}).get("registry", {})
                model_name: str = registry_conf.get("model_name", approach_cfg.model)
                version = "latest"

                model_uri = f"models:/{model_name}/{version}"
                print(f"[TestPipeline] Loading model from MLflow Registry: {model_uri}")
                return mlflow.pyfunc.load_model(model_uri)

            except Exception as e:
                print(
                    f"[TestPipeline] Warning: Could not load from MLflow ({e}). "
                    "Falling back to local artifacts."
                )

        print("[TestPipeline] Loading model from Local Artifacts...")

        name: str = approach_cfg.model.lower()
        hp: Dict[str, Any] = approach_cfg.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """
        Build the final model input window.

        Extracts the last rows equal to the configured sequence length.
        Adds a batch dimension and converts to float32.

        Args:
            df: Transformed DataFrame.
            input_chunk_length: Required sequence length.

        Returns:
            np.ndarray: Model input with shape [1, seq_len, n_features].

        Raises:
            ValueError: When df does not contain enough rows.
        """
        seq_len: int = input_chunk_length

        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")

        window: np.ndarray = df.iloc[-seq_len:].values
        return window[np.newaxis, :].astype(np.float32)

    def run_approach(self, approach: Dict[str, Any], data: pd.DataFrame) -> np.ndarray:
        """
        Execute inference for the given approach.

        Steps:
        1. Preprocess raw data.
        2. Build input window.
        3. Load model from MLflow or artifacts.
        4. Run forward prediction.

        Args:
            approach: Dict containing model name and hyperparameters.
            data: Raw historical DataFrame.

        Returns:
            np.ndarray: Model prediction output.
        """
        if not isinstance(approach, dict):
            raise TypeError(f"Expected approach to be dict, got {type(approach)}")

        df_transformed: pd.DataFrame = data
        input_chunk_length: int = approach.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )
        x_input: np.ndarray = self._prepare_input_window(
            df_transformed, input_chunk_length
        )

        wrapper: Any = self._load_model(approach)
        preds: np.ndarray = wrapper.predict(x_input)

        print(f"[INFERENCE] Input shape: {x_input.shape}, Output shape: {preds.shape}")
        print(f"[INFERENCE] Predictions: {preds.flatten()[:10]}...")

        return preds
