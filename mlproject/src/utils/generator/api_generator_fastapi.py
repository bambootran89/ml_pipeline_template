from __future__ import annotations

from typing import Any, Dict, List, Optional

from .api_generator_extractors import ApiGeneratorExtractorsMixin


class ApiGeneratorFastAPIMixin(ApiGeneratorExtractorsMixin):
    """FastAPI code generation mixin."""

    def _generate_fastapi_code(  # pylint: disable=too-many-locals
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate FastAPI code."""
        model_keys = [inf["model_key"] for inf in inference_steps]

        if data_config is None:
            data_config = {
                "data_type": "timeseries",
                "features": [],
                "target_columns": [],
                "input_chunk_length": 24,
                "output_chunk_length": 6,
            }

        data_type = data_config.get("data_type", "timeseries")
        features = data_config.get("features", [])
        input_chunk_length = data_config.get("input_chunk_length", 24)
        output_chunk_length = data_config.get("output_chunk_length", 6)

        # Build code using single quotes for inner docstrings
        code_parts = []

        # Header
        desc = (
            "Tabular batch prediction"
            if data_type == "tabular"
            else "Timeseries multi-step prediction"
        )
        api_desc = "tabular" if data_type == "tabular" else "timeseries"
        code_parts.append(
            f"""# Auto-generated FastAPI serve for {pipeline_name}
# Generated from serve configuration.
# Supports: {desc}

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
    title="{pipeline_name} API",
    version="1.0.0",
    description="Auto-generated serve API for {api_desc} data",
)


# Request/Response Schemas

class PredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(...,
        description="Input data as dict of columns to values"
    )


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input data with multiple rows")
    return_probabilities: bool = Field(
        default=False,
        description="Return prediction probabilities if available"
    )


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]] = Field(..., description="Input timeseries data")
    steps_ahead: int = Field(
        default={output_chunk_length},
        description="Number of steps to predict ahead",
        ge=1
    )


class PredictResponse(BaseModel):
    predictions: Dict[str, List[float]]
    metadata: Optional[Dict[str, Any]] = None

class MultiPredictResponse(BaseModel):
    predictions: Dict[str, List[List[float]]]
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_type: str = "{data_type}"
    features: List[str] = {features}


# Service Implementation

class ServeService:
    DATA_TYPE = "{data_type}"
    INPUT_CHUNK_LENGTH = {input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {output_chunk_length}
    FEATURES = {features}

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {{}}

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")
"""
        )

        # Load preprocessor
        preprocessor_artifact = self._get_preprocessor_artifact_name(
            preprocessor, load_map
        )
        if preprocessor_artifact:
            code_parts.append(
                f"""
            self.preprocessor = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{preprocessor_artifact}",
                alias="production",
            )
"""
            )

        # Load models
        for model_key in set(model_keys):
            step_id = load_map.get(model_key, "model")
            code_parts.append(
                f"""
            self.models["{model_key}"] = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{step_id}",
                alias="production",
            )
"""
            )

        # Service methods
        code_parts.append(
            """
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
            x_input = x_input[:self.INPUT_CHUNK_LENGTH, :]
            x_input = x_input.reshape(1, -1)
        else:
            x_input = x_input[:self.INPUT_CHUNK_LENGTH, :]
            x_input = x_input[np.newaxis, :]
        return x_input

    def predict_tabular_batch(
        self,
        context: Dict[str, Any],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        results = {}
        metadata = {"n_samples": 0, "model_type": "tabular"}
"""
        )

        # Tabular inference steps
        for step_info in inference_steps:
            code_parts.append(
                f"""
        model = self.models.get("{step_info['model_key']}")
        if model is not None:
            features = context.get("{step_info['features_key']}")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["{step_info['output_key']}"] = preds.flatten().tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        key = "{step_info['output_key']}_probabilities"
                        results[key] = proba.tolist()
                    except Exception:
                        pass
"""
            )

        code_parts.append(
            """
        return {"predictions": results, "metadata": metadata}

    def predict_timeseries_multistep(
        self,
        context: Dict[str, Any],
        steps_ahead: int
    ) -> Dict[str, Any]:
        results = {}
        n_blocks = (steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1)
        n_blocks = n_blocks // self.OUTPUT_CHUNK_LENGTH
        metadata = {
            "output_chunk_length": self.OUTPUT_CHUNK_LENGTH,
            "n_blocks": n_blocks,
            "model_type": "timeseries"
        }
"""
        )

        # Timeseries inference steps
        for step_info in inference_steps:
            code_parts.append(
                f"""
        model = self.models.get("{step_info['model_key']}")
        if model is not None:
            features = context.get("{step_info['features_key']}")
            if features is not None:
                all_predictions = []
                if isinstance(features, pd.DataFrame):
                    current_input = features.copy()
                else:
                    current_input = features
                for block_idx in range(n_blocks):
                    if len(current_input) < self.INPUT_CHUNK_LENGTH:
                        break
                    x_input = self._prepare_input_timeseries(
                        current_input,
                        "{step_info['model_type']}"
                    )
                    block_preds = model.predict(x_input)
                    # if hasattr(block_preds, "flatten"):
                    #     block_preds = block_preds.flatten()
                    # all_predictions.extend(block_preds.tolist())
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
                        if isinstance(current_input, pd.DataFrame):
                            current_input = current_input.iloc[shift:]
                preds_2d = np.concatenate(all_predictions, axis=0)
                results["{step_info['output_key']}"] = preds_2d.tolist()
"""
            )

        code_parts.append(
            """
        return {"predictions": results, "metadata": metadata}

    def run_inference_pipeline(self, context: Dict[str, Any]) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {})

"""
        )

        # Initialize service
        code_parts.append(
            f"""
service = ServeService("{experiment_config_path}")


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
        context = {{"preprocessed_data": preprocessed_data}}
        predictions = service.run_inference_pipeline(context)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

"""
        )

        # Add data-type specific endpoints
        if data_type == "tabular":
            code_parts.append(
                """
@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(request: BatchPredictRequest) -> PredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        result = service.predict_tabular_batch(
            context,
            return_probabilities=request.return_probabilities
        )
        return PredictResponse(
            predictions=result["predictions"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

"""
            )
        else:
            code_parts.append(
                """
@app.post("/predict/multistep", response_model=MultiPredictResponse)
def predict_multistep(request: MultiStepPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < service.INPUT_CHUNK_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Input must have at least \
                    {{service.INPUT_CHUNK_LENGTH}} timesteps (got {{len(df)}})"
            )
        preprocessed_data = service.preprocess(df)
        context = {{"preprocessed_data": preprocessed_data}}
        result = service.predict_timeseries_multistep(
            context,
            steps_ahead=request.steps_ahead
        )
        return MultiPredictResponse(
            predictions=result["predictions"],
            metadata=result["metadata"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

"""
            )

        code_parts.append(
            """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        )

        return "".join(code_parts)
