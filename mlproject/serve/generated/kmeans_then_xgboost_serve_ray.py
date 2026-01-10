"""Auto-generated Ray Serve deployment for kmeans_then_xgboost_serve.

Generated from serve configuration.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="kmeans_then_xgboost_serve Ray Serve API",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Prediction request."""
    data: Dict[str, List[Any]]


class PredictResponse(BaseModel):
    """Prediction response."""
    predictions: List[float]


class HealthResponse(BaseModel):
    """Health response."""
    status: str
    model_loaded: bool


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5})
class ModelService:
    """Model inference service."""

    def __init__(self, config_path: str) -> None:
        """Initialize model service."""
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models: Dict[str, Any] = {}
        self.ready = False

        self._load_models()

    def _load_models(self) -> None:
        """Load models from MLflow."""
        if not self.mlflow_manager.enabled:
            return

        experiment_name = self.cfg.experiment.get("name", "kmeans_then_xgboost_serve")

        # Load models
        self.models["fitted_kmeans_features"] = self.mlflow_manager.load_component(
            name=f"{experiment_name}_kmeans_features",
            alias="production",
        )
        self.models["fitted_xgboost_model"] = self.mlflow_manager.load_component(
            name=f"{experiment_name}_xgboost_model",
            alias="production",
        )

        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> None:
        """Health check."""
        if not self.ready:
            raise RuntimeError("ModelService not ready")

    async def predict(self, features: np.ndarray, model_key: str) -> List[float]:
        """Run inference."""
        model = self.models.get(model_key)
        if model is None:
            raise ValueError(f"Model {model_key} not found")

        preds = model.predict(features)
        return preds.flatten().tolist()

    def is_loaded(self) -> bool:
        """Check if models loaded."""
        return self.ready


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5})
class PreprocessService:
    """Preprocessing service."""

    def __init__(self, config_path: str) -> None:
        """Initialize preprocessing."""
        print("[PreprocessService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.ready = False

        self._load_preprocessor()

    def _load_preprocessor(self) -> None:
        """Load preprocessor from MLflow."""
        if not self.mlflow_manager.enabled:
            self.ready = True
            return

        experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")
        self.preprocessor = self.mlflow_manager.load_component(
            name=f"{experiment_name}_preprocessor",
            alias="production",
        )

        self.ready = True
        print("[PreprocessService] Ready")

    def check_health(self) -> None:
        """Health check."""
        if not self.ready:
            raise RuntimeError("PreprocessService not ready")

    async def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)


@serve.deployment
@serve.ingress(app)
class ServeAPI:
    """Main API gateway."""

    def __init__(
        self,
        preprocess_handle: Any,
        model_handle: Any,
    ) -> None:
        """Initialize API."""
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load(""mlproject/configs/pipelines/kmeans_then_xgboost.yaml"")
        self.input_chunk_length = self._get_input_chunk_length()

    def _get_input_chunk_length(self) -> int:
        """Get input chunk length from config."""
        if hasattr(self.cfg, "experiment") and hasattr(self.cfg.experiment, "hyperparams"):
            return int(self.cfg.experiment.hyperparams.get("input_chunk_length", 24))
        return 24

    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Prediction endpoint."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(request.data)

            # Preprocess
            features = await self.preprocess_handle.preprocess.remote(df)

            # Prepare input
            x_input = features.values[-self.input_chunk_length:]
            x_input = x_input[np.newaxis, :].astype(np.float32)

            # Predict
            preds = await self.model_handle.predict.remote(
                x_input, "fitted_kmeans_features"
            )

            return PredictResponse(predictions=preds)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        """Health check endpoint."""
        try:
            model_loaded = await self.model_handle.is_loaded.remote()
        except Exception:
            model_loaded = False

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
        )


def main() -> None:
    """Start Ray Serve application."""
    ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0")
    serve.start(detached=True)

    # Bind services
    config_path = "mlproject/configs/pipelines/kmeans_then_xgboost.yaml"
    model_service = ModelService.bind(config_path)
    preprocess_service = PreprocessService.bind(config_path)

    # Deploy API
    serve.run(
        ServeAPI.bind(preprocess_service, model_service),
        route_prefix="/",
    )

    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Press Ctrl+C to stop")

    import signal
    signal.pause()


if __name__ == "__main__":
    main()
