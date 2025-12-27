"""
TestPipeline: Simple inference pipeline for online serving.

Responsibilities:
- Load trained models from MLflow Registry or local artifacts.
- Preprocess raw input data for serving.
- Build input windows for time series models.
- Run forward prediction and return outputs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.dataio.feast_loader import FeastDatasetLoader
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class TestPipeline(BasePipeline):
    """
    Inference pipeline for online evaluation.

    This pipeline does not use DataModule, dataset splits, or batching logic.
    """

    def __init__(self, cfg_path: str = "", alias: str = "latest") -> None:
        """
        Initialize pipeline and load configuration.

        Args:
            cfg_path: Path to configuration file. If empty, default rules apply.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)

        self.model: Any = None
        self.preprocessor_model: Any | None = None

        if self.mlflow_manager.enabled:
            # self.model = self._load_model_from_mlflow()
            # Load artifacts đồng nhất
            model_name: str = self.cfg.experiment["model"].lower()
            self.preprocessor_model = self.mlflow_manager.load_component(
                name=f"{model_name}_preprocessor", alias=alias
            )
            self.model = self.mlflow_manager.load_component(
                name=f"{model_name}_model", alias=alias
            )

    def preprocess(self, data: Any = None) -> Any:
        """
        Preprocess input data for serving.

        Args:
            data: Optional raw input DataFrame.

        Returns:
            Preprocessed DataFrame or None if no data provided.
        """
        if self.preprocessor_model is not None:
            return self.preprocessor_model.transform(data)
        else:
            self.preprocessor.transform_manager.load(cfg=self.cfg)
            return self.preprocessor.transform_manager.transform(data)

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

    def _load_from_feast(self) -> pd.DataFrame:
        """
        Load latest features from Feast for real-time inference.

        This method is automatically called when:
        - No input data is provided to run_exp()
        - Config has data.path starting with "feast://"

        Returns:
            pd.DataFrame: Latest feature sequence from Feast online store.

        Raises:
            ValueError: If Feast URI is invalid or features unavailable.
        """
        feast_uri = self.cfg.data.get("path", "")
        features = self.cfg.data.get("features", [])

        if not feast_uri.startswith("feast://"):
            raise ValueError(
                "Feast URI required. Expected format: "
                "feast://repo?entity=key&id=val&features=..."
            )

        loader = FeastDatasetLoader()

        # Load features using same URI from config
        df = loader.load(
            path=feast_uri,
            index_col=self.cfg.data.get("index_col", "event_timestamp"),
            data_type="timeseries",
        )

        print(f"[Feast] Loaded {len(df)} rows from online store")
        return df[features]

    def run_exp(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Perform model inference using a fixed sequence of operations.

        Execution order:
        1) Load latest features from Feast online store if `data` is None.
        2) Apply preprocessing transformations.
        3) Construct an input window for the model.
        4) Run forward prediction using the loaded MLflow model.

        Args:
            data: Optional raw historical features. If None, the method attempts
                to pull the latest available features from Feast online store.

        Returns:
            Predictions as a NumPy array of shape (batch, horizon, ...).

        Raises:
            ValueError: If no data is supplied and no valid Feast URI is found in
                        the configuration.
        """
        # Step 1: Resolve data source
        if data is None:
            uri = self.cfg.data.get("path", "")
            logger.info("[INFERENCE] No data argument supplied, config.path='%s'", uri)

            if uri.startswith("feast://"):
                data = self._load_from_feast()
                logger.info(
                    "[INFERENCE] Feast source returned shape (%d, %d)",
                    data.shape[0],
                    data.shape[1],
                )
            else:
                raise ValueError(
                    "Inference requires either a DataFrame input or a Feast URI in "
                    "config.data.path starting with 'feast://'"
                )

        # Step 2: Preprocess raw features
        logger.info(
            "[INFERENCE] Running preprocessing on data with shape (%d, %d)",
            data.shape[0],
            data.shape[1],
        )
        df: pd.DataFrame = self.preprocess(data)

        # Step 3: Build model input window
        win: int = int(self.exp.get("hyperparams", {}).get("input_chunk_length", 24))
        logger.info("[INFERENCE] Building input window (length=%d)", win)

        x: np.ndarray = self._prepare_input_window(df, win)
        logger.info("[INFERENCE] Model input window prepared with shape %s", x.shape)

        # Step 4: Run model forward prediction
        logger.info("[INFERENCE] Calling model.predict()")
        y: np.ndarray = self.model.predict(x)

        # Output logging to user
        print(f"[INFERENCE] Input shape: {x.shape}, Output shape: {y.shape}")
        if hasattr(y, "flatten"):
            print(f"[INFERENCE] First 10 values: {y.flatten()[:10]}")

        logger.info("[INFERENCE] Inference completed")
        return y
