import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.base import BasePipeline
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request
from mlproject.src.tracking.mlflow_manager import MLflowManager

"""
TestPipeline module.

Provides a simple inference pipeline for online serving.
Does not use DataModule, dataset splits, or batching logic.
"""


class TestPipeline(BasePipeline):
    """
    Inference pipeline used for online evaluation.

    Responsibilities:
    - Load trained models from MLflow or local storage.
    - Preprocess raw input for serving.
    - Build input windows for time series models.
    - Run forward prediction and return model output.
    """

    def __init__(self, cfg_path: str = ""):
        """
        Initialize pipeline and load configuration.

        Creates MLflowManager when MLflow is enabled in
        configuration. Also initializes BasePipeline.

        Args:
            cfg_path: Path to configuration file. If empty,
                default loader rules apply.
        """
        self.cfg = ConfigLoader.load(cfg_path)
        super().__init__(self.cfg)
        self.mlflow_manager = MLflowManager(self.cfg)

    def preprocess(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Apply serving-time preprocessing to raw input.

        Loads the saved scaler and applies feature transforms.

        Args:
            data: Raw DataFrame containing historical input.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for
                model inference.

        Raises:
            ValueError: When no input data is provided.
        """
        if data is None:
            raise ValueError("Test mode requires input data")
        return serve_preprocess_request(data, self.cfg)

    def _load_model(self, approach: DictConfig):
        """
        Load a trained model based on configuration.

        Priority:
        1. MLflow Registry (when enabled).
        2. Local artifacts directory (fallback).

        Args:
            approach: Configuration block specifying the
                model name and hyperparameters.

        Returns:
            Any loaded model object (PyFunc or custom wrapper).
        """
        if self.mlflow_manager and self.mlflow_manager.enabled:
            try:
                registry_conf = self.cfg.get("mlflow", {}).get("registry", {})
                model_name = registry_conf.get("model_name", approach["model"])
                version = "latest"

                model_uri = f"models:/{model_name}/{version}"
                print(
                    "[TestPipeline] Loading model from MLflow " f"Registry: {model_uri}"
                )

                return mlflow.pyfunc.load_model(model_uri)

            except Exception as e:
                print(
                    "[TestPipeline] Warning: Could not load from MLflow "
                    f"({e}). Falling back to local artifacts."
                )

        print("[TestPipeline] Loading model from Local Artifacts...")

        name = approach["model"].lower()
        hp = approach.get("hyperparams", {})

        if isinstance(hp, DictConfig):
            hp = OmegaConf.to_container(hp, resolve=True)

        return ModelFactory.load(name, hp, self.cfg.training.artifacts_dir)

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """
        Build the final model input window.

        Extracts the last rows equal to the configured
        sequence length. Adds a batch dimension and converts
        to float32.

        Args:
            df: Transformed DataFrame.
            input_chunk_length: Required sequence length.

        Returns:
            np.ndarray: Model input with shape
                [1, seq_len, n_features].

        Raises:
            ValueError: When df does not contain enough rows.
        """
        seq_len = input_chunk_length

        if len(df) < seq_len:
            raise ValueError(
                f"Input data has {len(df)} rows, need at least " f"{seq_len}"
            )

        window = df.iloc[-seq_len:].values
        return window[np.newaxis, :].astype(np.float32)

    def run_approach(self, approach: DictConfig, data: pd.DataFrame) -> np.ndarray:
        """
        Execute inference for the given approach.

        Steps:
        1. Preprocess raw data.
        2. Build input window.
        3. Load model from MLflow or artifacts.
        4. Run forward prediction.

        Args:
            approach: Approach config containing the model
                name and hyperparameters.
            data: Raw historical DataFrame.

        Returns:
            np.ndarray: Model prediction output.
        """
        df_transformed = self.preprocess(data)

        input_chunk_length = approach.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )

        x_input = self._prepare_input_window(df_transformed, input_chunk_length)

        wrapper = self._load_model(approach)
        preds = wrapper.predict(x_input)

        print(
            f"[INFERENCE] Input shape: {x_input.shape}, " f"Output shape: {preds.shape}"
        )
        print(f"[INFERENCE] Predictions: {preds.flatten()[:10]}...")

        return preds
