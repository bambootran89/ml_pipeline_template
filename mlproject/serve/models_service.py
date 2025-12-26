"""ModelsService module.

Contains ModelsService which loads a model and runs inference.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlproject.serve.schemas import PredictRequest
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader


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
        if self.mlflow_manager.enabled:
            # self.model = self._load_model_from_mlflow()
            # Load artifacts đồng nhất
            model_name: str = self.cfg.experiment["model"].lower()
            self.preprocessor_model = self.mlflow_manager.load_component(
                name=f"{model_name}_preprocessor", alias="production"
            )
            self.model = self.mlflow_manager.load_component(
                name=f"{model_name}_model", alias="production"
            )

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
            return self.preprocessor_model.transform(data)

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
