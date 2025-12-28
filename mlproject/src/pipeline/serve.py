"""
TestPipeline: Simple inference pipeline for online serving.

Responsibilities:
- Load trained models from MLflow Registry or local artifacts.
- Preprocess raw input data for serving.
- Build input windows for time series models.
- Run forward prediction and return outputs.
"""

from __future__ import annotations

from typing import Any, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.src.features.factory import FeatureStoreFactory
from mlproject.src.features.timeseries import TimeSeriesFeatureStore
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.utils.config_loader import ConfigLoader


class TestPipeline(BasePipeline):
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
        Load data from Feast.

        - 'timeseries': Use TimeSeriesFeatureStore to
                retrieve a historical sequence window.
        - 'tabular': Fetch data directly from Online Store
                using get_online_features.

        Raises:
            ValueError: If URI is invalid or required config keys are missing.
        """
        # 1. Parse Feast URI
        uri: str = self.cfg.data.get("path", "")
        parsed = urlparse(uri)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"Invalid Feast URI: {uri}. Expected format 'feast://<repo_name>'"
            )

        repo_name: str = parsed.netloc
        data_type: str = self.cfg.data.get("type", "timeseries")

        if not hasattr(self.cfg.data, "featureview") or not hasattr(
            self.cfg.data, "features"
        ):
            raise ValueError("Missing 'featureview' or 'features' in cfg.data")

        # 2. Build list of feature references (view:feature)
        feature_refs: List[str] = [
            f"{self.cfg.data.featureview}:{f}" for f in self.cfg.data.features
        ]

        entity_key: str = self.cfg.data.get("entity_key", "location_id")
        entity_id: Any = self.cfg.data.get("entity_id", 1)

        # 3. Initialize Base Store
        base_store = FeatureStoreFactory.create(store_type="feast", repo_path=repo_name)

        # --- Tabular / Online Data Retrieval ---
        if data_type != "timeseries":
            print(
                f"[FEAST] Direct online fetch for \
                    tabular data (repo={repo_name}, type={data_type})"
            )

            online_data_list = base_store.get_online_features(
                entity_rows=[{entity_key: entity_id}],
                features=feature_refs,
            )

            if not online_data_list:
                raise ValueError(f"No online data found for {entity_key}={entity_id}")

            df = pd.DataFrame(online_data_list)
            return df[self.cfg.data.features]

        # --- Timeseries Data Retrieval ---
        else:
            print(f"[FEAST] Sequence fetch for timeseries data (repo={repo_name})")

            win_size: int = int(
                self.exp.get("hyperparams", {}).get("input_chunk_length", 24)
            )
            frequency_hours: int = int(self.cfg.data.get("frequency_hours", 1))

            ts_store = TimeSeriesFeatureStore(
                store=base_store,
                default_entity_key=entity_key,
                default_entity_id=entity_id,
            )

            # Retrieve sequence window
            df = ts_store.get_latest_n_sequence(
                features=feature_refs,
                n_points=win_size + (24 // frequency_hours),
                frequency_hours=frequency_hours,
                time_point=self.time_point,
            )

            # Fallback to cfg.data.end_date if no data
            if df.empty and hasattr(self.cfg.data, "end_date"):
                print(
                    f"[FEAST] No data at time_point={self.time_point}, "
                    f"falling back to cfg_end_date={self.cfg.data.end_date}"
                )
                df = ts_store.get_latest_n_sequence(
                    features=feature_refs,
                    n_points=win_size + (24 // frequency_hours),
                    frequency_hours=frequency_hours,
                    time_point=self.cfg.data.end_date,
                )

            if df.empty:
                raise ValueError("No data found in Feast for timeseries sequence.")

            index_col: str = self.cfg.data.get("index_col", "event_timestamp")
            df = df.set_index(index_col)[self.cfg.data.features]

            print(f"[FEAST] Loaded sequence window with {len(df)} rows")
            return df

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
            print(f"[INFERENCE] No data argument supplied, config.path={uri}")

            if uri.startswith("feast://"):
                data = self._load_from_feast()
                print(f"[INFERENCE] Feast source returned shape {data.shape}")
            else:
                raise ValueError(
                    "Inference requires either a DataFrame input or a Feast URI in "
                    "config.data.path starting with 'feast://'"
                )

        # Step 2: Preprocess raw features
        print(
            f"[INFERENCE] Running preprocessing on data with shape {data.shape}",
        )
        df: pd.DataFrame = self.preprocess(data)

        # Step 3: Build model input window
        win: int = int(self.exp.get("hyperparams", {}).get("input_chunk_length", 24))
        print("[INFERENCE] Building input window (length=%d)", win)

        x: np.ndarray = self._prepare_input_window(df, win)
        print(
            f"[INFERENCE] Model input window prepared with shape {x.shape}",
        )

        # Step 4: Run model forward prediction
        print("[INFERENCE] Calling model.predict()")
        y: np.ndarray = self.model.predict(x)

        # Output logging to user
        print(f"[INFERENCE] Input shape: {x.shape}, Output shape: {y.shape}")
        if hasattr(y, "flatten"):
            print(f"[INFERENCE] First 10 values: {y.flatten()[:10]}")

        print("[INFERENCE] Inference completed")
        return y
