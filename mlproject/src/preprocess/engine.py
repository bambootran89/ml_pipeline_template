from .base import PreprocessBase


class PreprocessEngine:
    """
    Singleton manager for preprocessing.

    Ensures that:
    - Scaler is loaded only once for online serving
    - Offline and online processing both use PreprocessBase
    - Prevents data drift between training and inference
    """

    _instance = None

    def __init__(self, cfg=None):
        """
        Initialize preprocessing engine.

        Args:
            cfg (dict, optional): Preprocessing configuration.
        """
        self.base = PreprocessBase(cfg)
        self._loaded = False

    @classmethod
    def instance(cls, cfg=None):
        """
        Return shared singleton instance.

        Args:
            cfg (dict, optional): Preprocessing configuration.

        Returns:
            PreprocessEngine: Shared instance.
        """
        if cls._instance is None:
            cls._instance = PreprocessEngine(cfg)
        return cls._instance

    def offline_fit(self, df):
        """
        Fit preprocessing pipeline on offline dataset.

        Args:
            df (pd.DataFrame): Raw DataFrame.

        Returns:
            pd.DataFrame: Processed DataFrame after fitting steps.
        """
        return self.base.fit(df)

    def offline_transform(self, df):
        """
        Transform dataset offline using fitted preprocessing steps.

        Args:
            df (pd.DataFrame)

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.base.transform(df)

    def online_transform(self, df):
        """
        Transform a single request online.

        Loads scaler lazily on first call.

        Args:
            df (pd.DataFrame): Raw request data.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not self._loaded:
            self.base.load_scaler()
            self._loaded = True

        return self.base.transform(df)
