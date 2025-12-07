import os

import numpy as np
import pandas as pd

from .base import ARTIFACT_DIR
from .engine import PreprocessEngine


class OfflinePreprocessor:
    """
    Offline data preprocessor for time series data.

    Handles missing value imputation, feature generation, scaling, and saving artifacts.

    Args:
        cfg (dict, optional): Configuration dictionary specifying preprocessing steps
                              and artifact directory.
    """

    def __init__(self, cfg=None):
        """
        Initialize OfflinePreprocessor.

        Args:
            cfg (dict, optional): Preprocessing configuration.
        """
        self.cfg = cfg or {}
        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])
        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.engine = PreprocessEngine.instance(cfg)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing pipeline on the DataFrame.

        Args:
            df (pd.DataFrame): Raw time series dataset.

        Returns:
            pd.DataFrame: Dataset after applying fitting steps.
        """
        return self.engine.offline_fit(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame using fitted preprocessing.

        Args:
            df (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        df = self.engine.offline_transform(df)
        self._save_features(df)
        return df

    def run(self) -> pd.DataFrame:
        """
        Execute full offline preprocessing pipeline:
        - load raw dataset
        - fit preprocessing
        - transform dataset
        - save features

        Returns:
            pd.DataFrame: Fully processed dataset.
        """
        df = self._load_raw_data()
        df = self.fit(df)
        df = self.transform(df)
        return df

    def _save_features(self, df):
        """
        Save transformed features to artifacts directory as Parquet.

        Args:
            df (pd.DataFrame): Processed dataset.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        df.to_parquet(os.path.join(self.artifacts_dir, "features.parquet"))

    def _load_raw_data(self):
        """
        Load raw dataset from CSV path provided in cfg.

        If the CSV is missing, generate synthetic time series.

        Returns:
            pd.DataFrame: Raw dataset.
        """
        data_cfg = self.cfg.get("data", {})
        path = data_cfg.get("path")
        index_col = data_cfg.get("index_col", "date")

        if not path or not os.path.exists(path):
            return self._load_synthetic(index_col)

        return self._load_csv(path, index_col)

    def _load_csv(self, path, index_col):
        """
        Load CSV file into DataFrame.

        Args:
            path (str): Path to CSV file.
            index_col (str): Column to treat as datetime index.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        df = pd.read_csv(path, parse_dates=[index_col])
        return df.set_index(index_col)

    def _load_synthetic(self, index_col):
        """
        Generate synthetic data when raw CSV is unavailable.

        Args:
            index_col (str): Name of index column.

        Returns:
            pd.DataFrame: Synthetic dataset.
        """
        idx = pd.date_range("2020-01-01", periods=200, freq="H")
        df = pd.DataFrame(
            {
                "HUFL": np.sin(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "MUFL": np.cos(np.arange(len(idx)) / 24)
                + np.random.randn(len(idx)) * 0.1,
                "mobility_inflow": np.random.rand(len(idx)) * 10,
            },
            index=idx,
        )
        df.index.name = index_col
        return df
