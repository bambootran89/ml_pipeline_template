class TSDataset:
    """
    Lightweight container for preprocessed time series data.

    Holds a precomputed DataFrame and provides train/validation/test splits.
    Does not compute heavy features; expects input DataFrame from offline pipeline.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        cfg (dict): Configuration dictionary containing split ratios under
                    cfg["preprocessing"]["split"].
    """

    def __init__(self, df, cfg):
        self.df = df.copy()
        # read split config
        sp = cfg.get("preprocessing", {}).get(
            "split", {"train": 0.6, "val": 0.2, "test": 0.2}
        )
        n = len(self.df)
        i_train = int(n * sp["train"])
        i_val = i_train + int(n * sp["val"])
        # store splits as pandas DataFrames
        self._train = self.df.iloc[:i_train]
        self._val = self.df.iloc[i_train:i_val]
        self._test = self.df.iloc[i_val:]

    def train(self):
        """
        Get the training split.

        Returns:
            pd.DataFrame: Training subset of the data.
        """
        return self._train

    def val(self):
        """
        Get the validation split.

        Returns:
            pd.DataFrame: Validation subset of the data.
        """
        return self._val

    def test(self):
        """
        Get the test split.

        Returns:
            pd.DataFrame: Test subset of the data.
        """
        return self._test
