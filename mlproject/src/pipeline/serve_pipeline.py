"""
TestPipeline: Simple inference pipeline for online serving.

Responsibilities:
- Load trained models from MLflow Registry or local artifacts.
- Preprocess raw input data for serving.
- Build input windows for time series models.
- Run forward prediction and return outputs.
"""

from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.online import OnlinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader


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
        self.preprocessor = OnlinePreprocessor(self.cfg)
        self.preprocessor.update_config(self.cfg)

    def preprocess(self, data: Any = None) -> Any:
        """
        Preprocess input data for serving.

        Args:
            data: Optional raw input DataFrame.

        Returns:
            Preprocessed DataFrame or None if no data provided.
        """
        if data is None:
            return None
        return self._preprocess_input(data)

    def _preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply serving-time preprocessing to raw input.

        Args:
            data: Raw DataFrame.

        Returns:
            Transformed DataFrame ready for inference.

        Raises:
            ValueError: If input data is None.
        """
        if data is None:
            raise ValueError("Test mode requires input data")
        return self.preprocessor.transform(data)

    def _load_model(self, approach: Dict[str, Any]) -> Any:
        """
        Load a trained model for inference.

        Priority:
        1. MLflow Registry if enabled.
        2. Local artifacts directory as fallback.

        Args:
            approach: Dictionary specifying model name and hyperparameters.

        Returns:
            Loaded model object.
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
                    f"[TestPipeline] Warning: MLflow load failed ({e}). "
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

        df_transformed: pd.DataFrame = self._preprocess_input(data)
        input_chunk_length: int = approach.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )
        x_input: np.ndarray = self._prepare_input_window(
            df_transformed, input_chunk_length
        )

        model_wrapper: Any = self._load_model(approach)
        preds: np.ndarray = model_wrapper.predict(x_input)

        print(f"[INFERENCE] Input shape: {x_input.shape}, Output shape: {preds.shape}")
        print(f"[INFERENCE] Predictions (first 10): {preds.flatten()[:10]}...")

        return preds
