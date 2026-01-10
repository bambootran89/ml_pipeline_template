"""Auto-generated FastAPI serve for standard_train_serve.

Generated from serve configuration.
"""

from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="standard_train_serve API",
    version="1.0.0",
    description="Auto-generated serve API",
)


class PredictRequest(BaseModel):
    """Prediction request schema."""
    data: Dict[str, List[Any]]


class PredictResponse(BaseModel):
    """Prediction response schema."""
    predictions: List[float]


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool


class ServeService:
    """Service for model inference."""

    def __init__(self, config_path: str) -> None:
        """Initialize service with config."""
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)

        # Load artifacts
        self.preprocessor = None
        self.models = {}

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "standard_train_serve")

            # Load preprocessor
            self.preprocessor = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocessor",
                alias="production",
            )

            # Load models
            self.models["fitted_train_model"] = self.mlflow_manager.load_component(
                name=f"{experiment_name}_train_model",
                alias="production",
            )

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data."""
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def predict(self, features: pd.DataFrame, model_key: str) -> List[float]:
        """Run model inference."""
        model = self.models.get(model_key)
        if model is None:
            raise RuntimeError(f"Model {model_key} not loaded")

        # Prepare input
        x_input = features.values[-24:]  # TODO: Get from config
        import numpy as np
        x_input = x_input[np.newaxis, :].astype(np.float32)

        # Predict
        preds = model.predict(x_input)
        return preds.flatten().tolist()


# Initialize service
service = ServeService("mlproject/configs/pipelines/standard_train.yaml")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    model_loaded = len(service.models) > 0
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Prediction endpoint."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        # Preprocess
        features = service.preprocess(df)

        # Predict with primary model
        predictions = service.predict(features, "fitted_train_model")

        return PredictResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
