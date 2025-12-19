"""ModelsService module.

Contains ModelsService which loads a model and runs inference.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.serve.schemas import PredictRequest
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.utils.mlflow_utils import (
    load_companion_preprocessor_from_model,
    load_model_from_registry_safe,
)


class ModelsService:
    """
    Unified service for loading models and serving predictions.
    """

    model: Optional[Any]
    preprocessor_model: Optional[Any]
    local_transform_manager: Optional[TransformManager]

    def __init__(self, cfg_path: str = "") -> None:
        """
        Initialize service using a config path.

        Parameters
        ----------
        cfg_path : str
            Path to the configuration YAML/JSON file.
        """
        self.cfg: DictConfig = ConfigLoader.load(cfg_path)
        self.model = None
        self.preprocessor_model = None
        self.local_transform_manager = None
        self.mlflow_manager = MLflowManager(self.cfg)
        self.load_model()

    def load_model(self) -> None:
        """Load prediction model and companion preprocessor."""
        if self.mlflow_manager.enabled:
            model = load_model_from_registry_safe(
                cfg=self.cfg,
                default_model_name=self.cfg.get("mlflow", {})
                .get("registry", {})
                .get("model_name", "ts_forecast_model"),
            )

            if model is not None:
                self.model = model
                self.preprocessor_model = load_companion_preprocessor_from_model(model)
                return

            print(
                "[ModelsService] MLflow load failed. Falling back to local artifacts."
            )

        self._load_model_from_local()

    def _load_model_from_local(self) -> None:
        """Load model from local artifacts."""
        if not self.cfg.get("approaches"):
            print("[ModelsService] No approaches configured.")
            return

        approach = self.cfg.approaches[0]
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        self.model = ModelFactory.load(
            name,
            hp,
            self.cfg.training.artifacts_dir,
        )

        print("[Service] Loading Local TransformManager...")
        self.local_transform_manager = TransformManager(
            self.cfg, artifacts_dir=self.cfg.training.artifacts_dir
        )
        if self.local_transform_manager is not None:
            try:
                self.local_transform_manager.load(self.cfg)
            except FileNotFoundError:
                print("[Service] Warning: Local preprocessing artifacts not found.")

    def _prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """Build model input window from preprocessed DataFrame."""
        input_chunk_length = 24
        if self.cfg.get("approaches"):
            input_chunk_length = self.cfg.approaches[0].hyperparams.get(
                "input_chunk_length", 24
            )

        if len(df) < input_chunk_length:
            raise ValueError(
                f"Not enough data. Needed {input_chunk_length}, got {len(df)}"
            )

        window = df.iloc[-input_chunk_length:].values
        return window[np.newaxis, :].astype(np.float32)

    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using the available preprocessing pipeline.

        Priority order:
        1. MLflow PyFunc preprocessor model (if available)
        2. Local offline preprocessor as a fallback

        Parameters
        ----------
        data : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Transformed feature data.
        """
        if self.preprocessor_model is not None:
            print("[Preprocess] Using MLflow PyFunc preprocessor.")
            return self.preprocessor_model.predict(data)

        if self.local_transform_manager is not None:
            print("[Preprocess] Using local preprocessor fallback.")
            return self.local_transform_manager.transform(data)

        raise RuntimeError("No preprocessing pipeline available.")

    def predict(self, request: PredictRequest) -> Dict[str, Any]:
        """
        Run full prediction pipeline and return JSON-serializable dict.

        Parameters
        ----------
        request : PredictRequest
            Input request containing raw data.

        Returns
        -------
        Dict[str, Any]
            Dictionary with key 'prediction' containing list of predictions.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        try:
            data = pd.DataFrame(request.data)
            df_transformed = self._transform_data(data)
            x_input = self._prepare_input_window(df_transformed)
            preds = self.model.predict(x_input)
            return {"prediction": preds.flatten().tolist()}

        except Exception as e:
            print(f"[ModelsService] Error during prediction: {e}")
            raise
