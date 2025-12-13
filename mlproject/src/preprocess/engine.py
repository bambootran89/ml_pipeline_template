from .base import PreprocessBase


class PreprocessEngine:
    """
    Singleton manager for preprocessing operations.

    This engine owns a shared `PreprocessBase` instance and handles
    configuration changes, lazy loading of preprocessing artifacts,
    and unified access to offline/online transformations. It is used
    across training, validation, and inference to ensure consistent
    preprocessing behavior.
    """

    def __init__(
        self,
        is_train,
        cfg=None,
    ):
        """
        Initialize the preprocessing engine.

        Parameters
        ----------
        is_train: is to load or train scaler
        cfg : dict | None
            Configuration dictionary. If None, an empty configuration
            is used to avoid initialization errors.
        """
        self._current_cfg = cfg or {}
        if is_train:
            self.base = PreprocessBase(self._current_cfg)
        else:
            self.base = None
            self.update_config(cfg)

    def update_config(
        self,
        cfg: dict,
    ):
        """
        Reload the underlying PreprocessBase if the MLflow run_id or
        preprocessing configuration changes.

        This ensures that API calls or inference services automatically
        switch to the correct preprocessing artifacts without requiring
        manual restarts.

        Parameters
        ----------
        new_cfg : dict
            Newly received configuration.
        """
        new_mlflow = cfg.get("mlflow", {})
        new_run_id = new_mlflow.get("run_id")
        print(
            f"""[PreprocessEngine] Configuration changed (Run ID:
                {new_run_id}). Reloading Base..."""
        )
        self._current_cfg = cfg
        self.base = PreprocessBase(self._current_cfg)
        self.base.load_scaler()
        print("[PreprocessEngine] Artifacts loaded successfully.")

    def offline_fit(self, df):
        """Fit preprocessing using offline/batch data."""
        return self.base.fit(df)

    def offline_transform(self, df):
        """Transform offline/batch data using the fitted preprocessing."""
        return self.base.transform(df)

    def online_transform(self, df):
        """
        Transform data in online/inference mode.

        The scaler is loaded lazily on first use, allowing inference
        services to initialize quickly and load preprocessing artifacts
        only when required.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data for online transformation.

        Returns
        -------
        pandas.DataFrame
            Transformed data after applying the loaded preprocessor.
        """

        return self.base.transform(df)
