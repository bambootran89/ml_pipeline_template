from typing import Optional

from .base import PreprocessBase
import threading


class PreprocessEngine:
    """
    Singleton manager for preprocessing operations.

    This engine owns a shared `PreprocessBase` instance and handles
    configuration changes, lazy loading of preprocessing artifacts,
    and unified access to offline/online transformations. It is used
    across training, validation, and inference to ensure consistent
    preprocessing behavior.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, cfg=None):
        """
        Initialize the preprocessing engine.

        Parameters
        ----------
        cfg : dict | None
            Configuration dictionary. If None, an empty configuration
            is used to avoid initialization errors.
        """
        self._current_cfg = cfg or {}
        self.base = PreprocessBase(self._current_cfg)
        self._loaded = False

    @classmethod
    def instance(cls, cfg: Optional[dict] = None):
        """
        Return the shared singleton instance.

        If an instance already exists and a new configuration is
        provided, the internal preprocessing base is reloaded when
        configuration differences require it (e.g., MLflow run_id changes).

        Parameters
        ----------
        cfg : dict | None
            Optional configuration update.

        Returns
        -------
        PreprocessEngine
            The global singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = PreprocessEngine(cfg)
        elif cfg is not None:
            cls._instance.update_config_if_needed(cfg)

        return cls._instance

    def update_config_if_needed(self, new_cfg: dict):
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
        current_mlflow = self._current_cfg.get("mlflow", {})
        new_mlflow = new_cfg.get("mlflow", {})

        current_run_id = current_mlflow.get("run_id")
        new_run_id = new_mlflow.get("run_id")

        should_reload = (new_run_id and new_run_id != current_run_id) or (
            not current_mlflow.get("enabled") and new_mlflow.get("enabled")
        )

        if should_reload:
            print(
                f"""[PreprocessEngine] Configuration changed (Run ID:
                  {new_run_id}). Reloading Base..."""
            )
            self._current_cfg = new_cfg
            self.base = PreprocessBase(self._current_cfg)
            self._loaded = False  # Force lazy load on next transform

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
        if not self._loaded:
            self.base.load_scaler()
            self._loaded = True

        return self.base.transform(df)
