from __future__ import annotations

import asyncio
import os
import signal
from typing import Any, Dict, List, Optional

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
from mlproject.src.utils.func_utils import get_env_path

ARTIFACTS_DIR: str = get_env_path(
    "ARTIFACTS_DIR",
    "mlproject/artifacts/models",
).as_posix()

CONFIG_PATH: str = get_env_path(
    "CONFIG_PATH",
    "mlproject/configs/experiments/etth1.yaml",
).as_posix()

app = FastAPI(title="mlproject Ray Serve Inference API")


@serve.deployment(
    health_check_period_s=10,
    health_check_timeout_s=30,
    num_replicas=int(os.getenv("RAY_PREPROCESS_REPLICAS", "2")),
    ray_actor_options={"num_cpus": 0.5},
)
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
    model_handle : Any
        Handle to ModelService for warmup.
    ready : bool
        Indicates whether preprocessing artifacts are loaded.
    """

    cfg: DictConfig
    mlflow_manager: MLflowManager
    preprocessor: Optional[Any]
    model_handle: Any
    ready: bool

    def __init__(self, model_handle: Any) -> None:
        """
        Initialize PreprocessingService and MLflow manager.

        Parameters
        ----------
        model_handle : Any
            Ray Serve handle to ModelService for warmup.
        """
        self.cfg = ConfigLoader.load(CONFIG_PATH)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.model_handle = model_handle
        self.ready = False

        # Trigger async warmup
        asyncio.create_task(self._warmup())

    async def _warmup(self) -> None:
        """
        Lazily initialize preprocessing artifacts.

        Priority:
        1. MLflow companion preprocessor (if available)
        2. Local TransformManager fallback
        """
        if self.ready:
            return

        if self.mlflow_manager.enabled:
            # self.model = self._load_model_from_mlflow()
            # Load artifacts đồng nhất
            model_name: str = self.cfg.experiment["model"].lower()
            self.preprocessor = self.mlflow_manager.load_component(
                f"{model_name}_preprocessor"
            )
            self.model = self.mlflow_manager.load_component(f"{model_name}_model")

        # Fallback: Load local TransformManager
        self.preprocessor = TransformManager(
            self.cfg,
            artifacts_dir=self.cfg.training.artifacts_dir,
        )
        self.preprocessor.load(cfg=self.cfg)
        self.ready = True
        print("[PreprocessingService] Loaded local TransformManager")

    def check_health(self) -> None:
        """Health check by Ray - allow warmup phase."""
        # Don't fail health check during warmup, just log status
        if not self.ready:
            print("[PreprocessingService] Still warming up...")
        # Health check passes even during warmup

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
            Ray actor handle to ModelService (unused, kept for compatibility).

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

    def check_health(self) -> None:
        """Health check by Ray."""
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
            model_name: str = self.cfg.experiment["model"].lower()
            self.model = self.mlflow_manager.load_component(f"{model_name}_model")
            if self.model is not None:
                self.model_loaded = True
            return

        self._load_local_model()
        return

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
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # PyFuncModel từ MLflow có method predict
        # Model local từ ModelFactory cũng có method predict
        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model has no predict method")

        preds = self.model.predict(x_input.astype(np.float32))
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
                req.data,
                self.model_handle,
            )  # type: ignore[attr-defined]

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
    # Start Ray với dashboard
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
    )
    serve.start(detached=True)

    # pylint: disable=no-member
    model = ModelService.bind()  # type: ignore[attr-defined]
    # pylint: disable=no-member
    preprocess = PreprocessingService.bind(model)  # type: ignore[attr-defined]

    # Deploy API với cả 2 handles
    # pylint: disable=no-member
    serve.run(
        ForecastAPI.bind(preprocess, model),  # type: ignore[attr-defined]
        route_prefix="/",
    )

    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Dashboard at http://localhost:8265")
    print("[Ray Serve] Press Ctrl+C to stop")

    # Giữ script chạy
    try:
        signal.pause()  # Wait for signal (Ctrl+C)
    except KeyboardInterrupt:
        print("\n[Ray Serve] Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
