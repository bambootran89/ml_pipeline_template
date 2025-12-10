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

"""
ModelsService module.

Contains ModelsService which loads a model and runs inference.
"""


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
        """
        Load model according to configuration priority.

        Priority:
        1. MLflow Registry (if enabled).
        2. Local artifacts directory (fallback).

        The method assigns self.model. On failure to load
        from MLflow, it logs a warning and loads local model.
        """
        # 1. Try Loading from MLflow Registry
        if self.mlflow_manager.enabled:
            try:
                registry_conf = self.cfg.get("mlflow", {}).get("registry", {})
                model_name = registry_conf.get("model_name", "ts_forecast_model")
                version = "latest"  # Can be parameterized if needed

                model_uri = f"models:/{model_name}/{version}"
                print(
                    f"[ModelsService] Loading model from MLflow Registry: "
                    f"{model_uri}"
                )

                self.model = mlflow.pyfunc.load_model(model_uri)
                return
            except Exception as e:  # pragma: no cover - runtime loading error
                print(
                    "[ModelsService] Warning: Could not load from MLflow "
                    f"({e}). Falling back to local artifacts."
                )

        # 2. Fallback: Load from Local Artifacts
        print("[ModelsService] Loading model from Local Artifacts...")

        # Assume single approach for simplicity or get from config
        approach = self.cfg.approaches[0]
        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        self.model = ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def _prepare_input_window(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build the model input window from a preprocessed DataFrame.

        Steps:
        - Read input_chunk_length from config.
        - Ensure DataFrame has enough rows.
        - Extract the last window rows.
        - Return array shaped [1, seq_len, features] as float32.

        Raises:
            ValueError: If provided DataFrame is shorter than
                the configured input window length.
        """
        # Get input_chunk_length from first approach hyperparams
        input_chunk_length = self.cfg.approaches[0].hyperparams.get(
            "input_chunk_length", 24
        )

        if len(df) < input_chunk_length:
            raise ValueError(
                f"Not enough data. Needed {input_chunk_length}, got {len(df)}"
            )

        # Take the last 'input_chunk_length' rows
        window = df.iloc[-input_chunk_length:].values

        # Expand dims to [1, seq_len, features] and cast to float32
        # (Required for MLflow PyFunc and many wrappers)
        return window[np.newaxis, :].astype(np.float32)

    def predict(self, request: PredictRequest) -> Dict[str, Any]:
        """
        Run full prediction pipeline and return JSON-serializable dict.

        Flow:
        1. Convert request.data to pandas.DataFrame.
        2. Apply serving preprocessing (serve_preprocess_request).
        3. Build input window via _prepare_input_window.
        4. Call model.predict and flatten results.

        Args:
            request: PredictRequest object containing input data.
                Expectation: request.data is convertible to a
                pandas.DataFrame (list[dict] or list[list]).

        Returns:
            A dict with key "prediction" and value as a list of
            numeric predictions (JSON serializable).

        Raises:
            RuntimeError: If model is not loaded.
            Exception: Propagates unexpected errors during processing.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        try:
            # 1. Convert Input Data to DataFrame
            # Assuming request.data is a list of dicts or list of lists
            data = pd.DataFrame(request.data)

            # 2. Preprocess (Scaling/Feature Engineering)
            df_transformed = serve_preprocess_request(data, self.cfg)

            # 3. Prepare Input Window
            x_input = self._prepare_input_window(df_transformed)

            # 4. Predict
            # Works for both MLflow PyFunc and Custom Wrapper
            preds = self.model.predict(x_input)

            # 5. Format Output
            # Flatten to 1D list for JSON response
            return {"prediction": preds.flatten().tolist()}

        except Exception as e:  # pragma: no cover - propagate runtime errors
            print(f"[ModelsService] Error during prediction: {e}")
            raise e
