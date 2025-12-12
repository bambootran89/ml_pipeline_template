from typing import Any, Dict, List

import pandas as pd


class TimeSeriesFoldSplitter:
    """
    Split a time-series DataFrame into contiguous folds while respecting
    model input/output window lengths.

    This splitter is useful for expanding-window or rolling-origin
    cross-validation where each fold must contain at least:

    - `input_chunk_length` samples for model input
    - `output_chunk_length` samples reserved for forecasting target

    Attributes:
        cfg: Full experiment configuration as a plain dictionary.
        df_cfg: Sub-config for dataset loading (path, index column, etc.).
        experiment_cfg: Sub-config for experiment hyperparameters.
        input_len: Required history size for model input.
        output_len: Required forecast length.
        n_splits: Number of desired folds.
        df: Loaded DataFrame (after parsing and sorting).
        n_samples: Total number of rows in the time series.
    """

    def __init__(self, cfg: Dict[str, Any], n_splits: int) -> None:
        """
        Initialize the splitter from configuration and number of folds.

        Args:
            cfg: Dictionary-like configuration object.
            n_splits: Number of folds to generate.
        """
        self.cfg = cfg
        self.df_cfg = cfg.get("data", {})
        self.experiment_cfg = cfg.get("experiment", {})

        hp_cfg = self.experiment_cfg.get("hyperparams", {})
        self.input_len = int(hp_cfg.get("input_chunk_length", 24))
        self.output_len = int(hp_cfg.get("output_chunk_length", 6))

        self.n_splits = int(n_splits)

        self.df: pd.DataFrame | None = None
        self.n_samples = 0

        self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Load the DataFrame from disk, parse datetime index, and sort.

        Returns:
            The loaded and indexed DataFrame.

        Raises:
            FileNotFoundError: If the CSV file path is invalid.
            ValueError: If index column is missing.
        """
        path = self.df_cfg.get("path")
        index_col = self.df_cfg.get("index_col")

        if path is None:
            raise ValueError("Dataset config missing required field: 'path'.")
        if index_col is None:
            raise ValueError("Dataset config missing required field: 'index_col'.")

        df = pd.read_csv(path, parse_dates=[index_col])
        df = df.set_index(index_col).sort_index()

        self.df = df
        self.n_samples = len(df)
        return df

    def generate_folds(self) -> List[pd.DataFrame]:
        """
        Generate contiguous DataFrame folds for time-series CV.

        Each fold expands the training window (expanding-window style)
        and guarantees at least ``input_len`` samples and room for an
        ``output_len``-sized forecast horizon.

        Returns:
            A list of DataFrames, each representing one fold.

        Raises:
            RuntimeError: If data is not loaded before folding.
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call `_load_data()` first.")

        min_train_size = self.input_len
        max_train_end = self.n_samples - self.output_len

        if max_train_end <= min_train_size:
            raise ValueError(
                "Dataset too small for given input/output lengths "
                f"(input={self.input_len}, output={self.output_len})."
            )

        step = max(1, (max_train_end - min_train_size) // self.n_splits)

        folds: List[pd.DataFrame] = []

        for i in range(self.n_splits):
            fold_start = 0
            fold_end = min_train_size + step * (i + 1)

            # Pylint R1730: use `min(...)`
            fold_end = min(fold_end, self.n_samples)

            fold_df = self.df.iloc[fold_start:fold_end].copy()
            folds.append(fold_df)

            if fold_end == self.n_samples:
                break

        return folds
