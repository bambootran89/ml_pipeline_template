# Auto-generated FastAPI serve for standard_train_serve
# Generated from serve configuration.
# Supports: Tabular batch prediction

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="standard_train_serve API",
    version="1.0.0",
    description="Auto-generated serve API for tabular data",
)


# Request/Response Schemas


class PredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(
        ..., description="Input data as dict of columns to values"
    )


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input data with multiple rows")
    return_probabilities: bool = Field(
        default=False, description="Return prediction probabilities if available"
    )


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input timeseries data")
    steps_ahead: int = Field(
        default=6, description="Number of steps to predict ahead", ge=1
    )


class PredictResponse(BaseModel):
    predictions: Dict[str, List[float]]
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_type: str = "tabular"
    features: List[str] = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]


# Service Implementation


class ServeService:
    DATA_TYPE = "tabular"
    INPUT_CHUNK_LENGTH = 24
    OUTPUT_CHUNK_LENGTH = 6
    FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {}

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "standard_train_serve")

            self.preprocessor = self.mlflow_manager.load_component(
                name=f"{experiment_name}_preprocess",
                alias="production",
            )

            self.models["fitted_train_model"] = self.mlflow_manager.load_component(
                name=f"{experiment_name}_train_model",
                alias="production",
            )

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def _prepare_input_timeseries(self, features: Any, model_type: str) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)

        if model_type == "ml":
            if x_input.ndim == 2:
                x_input = x_input.reshape(1, -1)
            elif x_input.ndim == 3:
                x_input = x_input.reshape(x_input.shape[0], -1)
        else:
            if x_input.ndim == 2:
                x_input = x_input[np.newaxis, :]
        return x_input

    def predict_tabular_batch(
        self, context: Dict[str, Any], return_probabilities: bool = False
    ) -> Dict[str, Any]:
        results = {}
        metadata = {"n_samples": 0, "model_type": "tabular"}

        model = self.models.get("fitted_train_model")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["train_model_predictions"] = preds.flatten().tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        results[
                            "train_model_predictions_probabilities"
                        ] = proba.tolist()
                    except Exception:
                        pass

        return {"predictions": results, "metadata": metadata}

    def predict_timeseries_multistep(
        self, context: Dict[str, Any], steps_ahead: int
    ) -> Dict[str, Any]:
        results = {}
        n_blocks = (
            steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1
        ) // self.OUTPUT_CHUNK_LENGTH
        metadata = {
            "steps_ahead": steps_ahead,
            "output_chunk_length": self.OUTPUT_CHUNK_LENGTH,
            "n_blocks": n_blocks,
            "model_type": "timeseries",
        }

        model = self.models.get("fitted_train_model")
        if model is not None:
            features = context.get("preprocessed_data")
            if features is not None:
                all_predictions = []
                current_input = (
                    features.copy() if isinstance(features, pd.DataFrame) else features
                )
                for block_idx in range(n_blocks):
                    x_input = self._prepare_input_timeseries(current_input, "ml")
                    block_preds = model.predict(x_input)
                    if hasattr(block_preds, "flatten"):
                        block_preds = block_preds.flatten()
                    all_predictions.extend(block_preds.tolist())
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
                        if isinstance(current_input, pd.DataFrame):
                            current_input = current_input.iloc[shift:]
                results["train_model_predictions"] = all_predictions[:steps_ahead]

        return {"predictions": results, "metadata": metadata}

    def run_inference_pipeline(self, context: Dict[str, Any]) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context, steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {})


service = ServeService("mlproject/configs/experiments/tabular.yaml")


# API Endpoints


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = len(service.models) > 0
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        data_type=service.DATA_TYPE,
        features=service.FEATURES,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        predictions = service.run_inference_pipeline(context)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(request: BatchPredictRequest) -> PredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        result = service.predict_tabular_batch(
            context, return_probabilities=request.return_probabilities
        )
        return PredictResponse(
            predictions=result["predictions"], metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
