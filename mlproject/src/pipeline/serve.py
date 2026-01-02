"""
ServingPipeline: Simple inference pipeline for online serving.

Responsibilities:
- Load trained models from MLflow Registry or local artifacts.
- Preprocess raw input data for serving.
- Build input windows for time series models.
- Run forward prediction and return outputs.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.utils.config_loader import ConfigLoader


class ServingPipeline(BasePipeline):
    """
    Inference pipeline for online evaluation.

    This pipeline does not use DataModule, dataset splits, or batching logic.
    """

    def __init__(
        self, cfg_path: str = "", alias: str = "latest", time_point: str = "now"
    ) -> None:
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
        self.time_point = time_point

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
        Load features from Feast using unified facade.

        Returns
        -------
        pd.DataFrame
            Features ready for preprocessing.
            - Tabular: Single row from Online Store
            - Timeseries: Indexed sequence window

        Raises
        ------
        ValueError
            If Feast URI is invalid or data loading fails.
        """
        facade = FeatureStoreFacade(self.cfg, mode="online")
        return facade.load_features(time_point=self.time_point)

    def run_exp(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Run model inference.

        Steps:
        1) Load features from Feast if data is None.
        2) Apply preprocessing transformations.
        3) Construct input window for timeseries, or use full DataFrame for tabular.
        4) Run forward prediction.

        Args:
            data: Optional raw features for inference.

        Returns:
            Model predictions as np.ndarray.
        """
        if data is None:
            print("[INFERENCE] No input DataFrame, loading from Feast...")
            data = self._load_from_feast()
            print(f"[INFERENCE] Loaded data shape: {data.shape}")

        print(f"[INFERENCE] Preprocessing data with shape {data.shape}")
        df: pd.DataFrame = self.preprocess(data)

        # --- Handle timeseries vs tabular separately ---
        data_type: str = self.cfg.data.get("type", "timeseries")
        if data_type == "timeseries":
            entity_key: str = self.cfg.data.get("entity_key", "location_id")
            win: int = int(
                self.cfg.experiment.get("hyperparams", {}).get("input_chunk_length", 24)
            )
            if "entity_key" in df.columns:
                arr_list: List[np.ndarray] = [
                    self._prepare_input_window(
                        g.drop(columns=[entity_key], errors="ignore"),
                        win,
                    )
                    for _, g in df.groupby(entity_key)
                ]
                print(f"[INFERENCE] Building input window of length {win}")

                x = np.vstack(arr_list).astype(np.float32)

                print(f"[INFERENCE] Input window shape: {x.shape}")
            else:
                x = self._prepare_input_window(
                    df,
                    win,
                ).astype(np.float32)

        else:
            # For tabular, no sequence window; use full preprocessed DataFrame
            print("[INFERENCE] Tabular input, using full DataFrame")
            x = df.values.astype(np.float32)
            print(f"[INFERENCE] Input array shape: {x.shape}")

        print("[INFERENCE] Running model.predict()")
        y: np.ndarray = self.model.predict(x)
        print(f"[INFERENCE] Output shape: {y.shape}")

        if hasattr(y, "flatten"):
            print(f"[INFERENCE] First 10 output values: {y[:10]}")

        print("[INFERENCE] Inference completed")
        return y
