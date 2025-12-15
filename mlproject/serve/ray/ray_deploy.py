from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from ray import serve

from mlproject.serve.schemas import PredictRequest
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.preprocess.transform_manager import TransformManager
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_loader import ConfigLoader
from mlproject.src.utils.mlflow_utils import load_model_from_registry_safe

ARTIFACTS_DIR = os.path.join("mlproject", "artifacts", "models")
CONFIG_PATH = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")

app = FastAPI(title="mlproject Ray Serve Inference API")


@serve.deployment(health_check_period_s=10, health_check_timeout_s=30)
class PreprocessingService:
    """
    Ray Serve deployment for input preprocessing.

    Attributes
    ----------
    cfg : DictConfig
        Experiment configuration.
    mlflow_manager : MLflowManager
        MLflow manager instance.
    preprocessor : Optional[Any]
        Either a PyFunc preprocessor or TransformManager.
    initialized : bool
        Indicates whether preprocessing artifacts are loaded.
    """

    cfg: DictConfig
    mlflow_manager: MLflowManager
    preprocessor: Optional[Any]
    ready: bool

    def __init__(self, model_handle: Any) -> None:
        """Initialize PreprocessingService and MLflow manager."""
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None

        self.model_handle = model_handle
        self.ready = False

        asyncio.create_task(self._warmup())

    async def _warmup(self) -> None:
        """
        Lazily initialize preprocessing artifacts.

        Priority:
        1. MLflow companion preprocessor (if available)
        2. Local TransformManager fallback

        Parameters
        ----------
        model_handle : Any
            Ray actor handle to ModelService.
        """
        if self.ready:
            return

        run_id: Optional[str] = None
        while run_id is None:
            try:
                run_id = await self.model_handle.get_run_id.remote()
            except Exception:
                # Model might not be initialized yet, wait before retry
                await asyncio.sleep(1)

        print(
            f"[PreprocessingService] Retrieved Run ID: {run_id}. Loading artifacts..."
        )

        if self.mlflow_manager.enabled and run_id:
            try:
                pp_uri = f"runs:/{run_id}/preprocessing_pipeline"
                self.preprocessor = mlflow.pyfunc.load_model(pp_uri)
                self.ready = True
                self.ready = True
                return
            except Exception as exc:
                print(
                    f"[PreprocessingService] Companion preprocessor load failed: {exc}"
                )

        # Load local TransformManager
        self.preprocessor = TransformManager(
            artifacts_dir=self.cfg.training.artifacts_dir,
        )
        self.preprocessor.load(cfg=self.cfg)
        self.ready = True

    def check_health(self):
        """
        check health by ray
        """
        if not self.ready:
            raise RuntimeError("PreprocessingService is still warming up...")

    async def preprocess(
        self, data: Dict[str, List[Any]], model_handle: Any
    ) -> pd.DataFrame:
        """
        Transform raw input payload into feature DataFrame.

        Parameters
        ----------
        data : Dict[str, List[Any]]
            Raw input payload from HTTP request.
        model_handle : Any
            Ray actor handle to ModelService.

        Returns
        -------
        pd.DataFrame
            Transformed feature DataFrame.
        """
        _ = model_handle
        if not self.ready:
            raise RuntimeError("Service temporarily unavailable")

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df = df.set_index("date")

        loop = asyncio.get_running_loop()
        assert self.preprocessor is not None
        return await loop.run_in_executor(None, self.preprocessor.transform, df)


@serve.deployment(health_check_period_s=10, health_check_timeout_s=30)
class ModelService:
    """
    Ray Serve deployment for model inference.

    Attributes
    ----------
    cfg : DictConfig
        Experiment configuration.
    mlflow_manager : MLflowManager
        MLflow manager instance.
    model : Optional[Any]
        Loaded model instance.
    model_loaded : bool
        Indicates if the model is loaded.
    run_id : Optional[str]
        MLflow run ID if loaded from MLflow.
    input_chunk_length : int
        Number of time steps for model input.
    """

    cfg: DictConfig
    mlflow_manager: MLflowManager
    model: Optional[Any]
    model_loaded: bool
    run_id: Optional[str]
    input_chunk_length: int

    def __init__(self) -> None:
        """Initialize ModelService, load MLflow manager and model."""
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.model = None
        self.model_loaded = False
        self.run_id = None
        self.input_chunk_length = self._load_input_chunk_length()
        self._load_model()
        print(f"[ModelService] Initialization Complete. Loaded: {self.model_loaded}")

    def check_health(self):
        """check_health"""
        if not self.model_loaded:
            raise RuntimeError("Model artifacts not loaded yet.")

    def _load_input_chunk_length(self) -> int:
        """
        Load input_chunk_length from experiment config.

        Returns
        -------
        int
            Number of input time steps.
        """
        if hasattr(self.cfg, "experiment") and hasattr(
            self.cfg.experiment, "hyperparams"
        ):
            return int(self.cfg.experiment.hyperparams.get("input_chunk_length", 24))
        return 24

    def _load_model(self) -> None:
        """Load model from MLflow or fallback to local artifacts."""
        if self.mlflow_manager.enabled:
            self._try_load_mlflow_model()
        if not self.model_loaded:
            self._load_local_model()

    def _try_load_mlflow_model(self) -> None:
        """Attempt loading model from MLflow registry."""
        result = load_model_from_registry_safe(
            self.cfg, default_model_name="ts_forecast_model"
        )
        if result is None:
            return
        self.model = result.model
        self.run_id = result.run_id
        self.model_loaded = True

    def _load_local_model(self) -> None:
        """Load model from local artifacts directory."""
        approach = self.cfg.approaches[0]
        name = approach["model"].lower()
        hyperparams = approach.get("hyperparams", {})
        if isinstance(hyperparams, DictConfig):
            hyperparams = OmegaConf.to_container(hyperparams, resolve=True)
        self.model = ModelFactory.load(name, hyperparams, ARTIFACTS_DIR)
        self.model_loaded = True

    def prepare_input(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare model input from preprocessed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed feature DataFrame.

        Returns
        -------
        np.ndarray
            2D array with shape (1, input_chunk_length, n_features).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if len(df) < self.input_chunk_length:
            raise ValueError(
                f"Need at least {self.input_chunk_length} rows, got {len(df)}"
            )
        return df.values[-self.input_chunk_length :][np.newaxis, :]

    def predict_prepared(self, x_input: np.ndarray) -> List[float]:
        """
        Run prediction on prepared input array.

        Parameters
        ----------
        x_input : np.ndarray
            Prepared input array.

        Returns
        -------
        List[float]
            Flattened prediction results.
        """
        if self.model is None or not hasattr(self.model, "predict"):
            raise RuntimeError("Model not loaded or has no predict method")
        preds = self.model.predict(x_input)
        return np.asarray(preds).flatten().tolist()

    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""
        return self.model_loaded

    def get_run_id(self) -> Optional[str]:
        """Return MLflow run ID."""
        return self.run_id

    def get_model(self) -> Optional[Any]:
        """Return underlying model instance."""
        return self.model


@serve.deployment
@serve.ingress(app)
class ForecastAPI:
    """
    Ray Serve HTTP API for model-agnostic inference.

    Attributes
    ----------
    preprocess_handle : Any
        Ray actor handle for PreprocessingService.
    model_handle : Any
        Ray actor handle for ModelService.
    """

    preprocess_handle: Any
    model_handle: Any

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        """
        Initialize ForecastAPI with preprocessing and model handles.

        Parameters
        ----------
        preprocess_handle : Any
            Preprocessing Ray actor handle.
        model_handle : Any
            Model Ray actor handle.
        """
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle

    @app.post("/predict")
    async def predict(self, req: PredictRequest) -> Dict[str, List[float]]:
        """
        Run model prediction for input payload.

        Parameters
        ----------
        req : PredictRequest
            FastAPI request payload containing data.

        Returns
        -------
        Dict[str, List[float]]
            Dictionary with prediction results.
        """
        try:
            # Call .remote() on Ray handles
            df: pd.DataFrame = await self.preprocess_handle.preprocess.remote(
                # type: ignore[attr-defined]
                req.data,
                self.model_handle,
            )
            # type: ignore[attr-defined]
            x_input = await self.model_handle.prepare_input.remote(df)
            # type: ignore[attr-defined]
            preds = await self.model_handle.predict_prepared.remote(x_input)
            return {"prediction": preds}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {exc}"
            ) from exc

    @app.get("/health")
    async def health(self) -> Dict[str, bool]:
        """
        Health check endpoint.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating API and model status.
        """
        try:
            # type: ignore[attr-defined]
            loaded = await self.model_handle.is_loaded.remote()
        except Exception:
            loaded = False
        return {"status": True, "model_loaded": loaded}


def main() -> None:
    """
    Start Ray Serve application.

    Initializes Ray, starts Serve, and deploys ForecastAPI.
    """
    ray.init(ignore_reinit_error=True)
    serve.start(detached=True)

    # pylint: disable=no-member

    preprocess = PreprocessingService.bind()  # type: ignore[attr-defined]

    # pylint: disable=no-member

    model = ModelService.bind()  # type: ignore[attr-defined]

    serve.run(
        # pylint: disable=no-member
        ForecastAPI.bind(preprocess, model),  # type: ignore[attr-defined]
        route_prefix="/",
    )

    print("[Ray Serve] API ready at http://localhost:8000")


if __name__ == "__main__":
    main()
