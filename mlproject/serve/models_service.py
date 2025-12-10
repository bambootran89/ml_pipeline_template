"""ModelsService module.

Contains ModelsService which loads a model and runs inference.
"""

from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.serve.schemas import PredictRequest
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request
from mlproject.src.tracking.mlflow_manager import MLflowManager


class ModelsService:
    """
    Unified service for loading models and serving predictions.

    Responsibilities:
    - Load model from MLflow Registry when enabled.
    - Fall back to local artifacts when registry access fails.
    - Preprocess incoming data for serving.
    - Build model input windows and run inference.
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize service using a config path.

        Loads configuration, instantiates MLflow manager,
        and attempts to load the model on initialization.

        Args:
            cfg_path: Path to configuration file or empty to
                use default loader behavior.
        """
        self.cfg = ConfigLoader.load(cfg_path)
        self.model = None
        self.mlflow_manager = MLflowManager(self.cfg)

        # Load model on initialization
        self.load_model()

    def load_model(self):
        """Load model from MLflow Registry or fallback to local artifacts."""
        if self.mlflow_manager.enabled:
            try:
                registry_conf = self.cfg.get("mlflow", {}).get("registry", {})
                model_name = registry_conf.get("model_name", "ts_forecast_model")
                version = "latest"
                model_uri = f"models:/{model_name}/{version}"
                print(
                    f"[ModelsService] Loading model from MLflow Registry: {model_uri}"
                )
                self.model = mlflow.pyfunc.load_model(model_uri)
                return
            except Exception as e:
                print(
                    f"[ModelsService] Warning: Could not load from MLflow ({e}). "
                    "Falling back to local artifacts."
                )

        # Fallback to local artifacts
        print("[ModelsService] Loading model from Local Artifacts...")
        approach = self.cfg.approaches[0]
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        self.model = ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def _prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """Build model input window from preprocessed DataFrame."""
        input_chunk_length = self.cfg.approaches[0].hyperparams.get(
            "input_chunk_length", 24
        )
        if len(df) < input_chunk_length:
            raise ValueError(
                f"Not enough data. Needed {input_chunk_length}, got {len(df)}"
            )

        window = df.iloc[-input_chunk_length:].values
        return window[np.newaxis, :].astype(np.float32)

    def predict(self, request: PredictRequest) -> Dict[str, Any]:
        """Run full prediction pipeline and return JSON-serializable dict."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        try:
            data = pd.DataFrame(request.data)
            df_transformed = serve_preprocess_request(data, self.cfg)
            x_input = self._prepare_input_window(df_transformed)
            preds = self.model.predict(x_input)
            return {"prediction": preds.flatten().tolist()}

        except Exception as e:
            print(f"[ModelsService] Error during prediction: {e}")
            raise e
