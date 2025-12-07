import pandas as pd


class BaseDataModule:
    """Base class for ML / DL data modules."""

    def __init__(self, df: pd.DataFrame, cfg: dict, target_column: str):
        self.df: pd.DataFrame = df
        self.cfg: dict = cfg
        self.target_column: str = target_column

        # DataFrame splits
        self.train_df: pd.DataFrame
        self.val_df: pd.DataFrame
        self.test_df: pd.DataFrame

        self._split_data()

    def _split_data(self):
        sp = self.cfg.get("preprocessing", {}).get(
            "split", {"train": 0.6, "val": 0.2, "test": 0.2}
        )
        n = len(self.df)
        i_train = int(n * sp["train"])
        i_val = i_train + int(n * sp["val"])
        self.train_df = self.df.iloc[:i_train]
        self.val_df = self.df.iloc[i_train:i_val]
        self.test_df = self.df.iloc[i_val:]
