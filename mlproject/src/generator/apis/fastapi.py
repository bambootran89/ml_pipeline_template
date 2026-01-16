from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import ApiGeneratorExtractorsMixin
from .types import DataConfig, GenerationContext


class ApiGeneratorFastAPIMixin(ApiGeneratorExtractorsMixin):
    """FastAPI code generation mixin."""

    def _generate_fastapi_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]] = None,
        alias: str = "production",
    ) -> str:
        """Generate FastAPI code."""
        ctx = self._create_generation_context(
            pipeline_name,
            load_map,
            preprocessor,
            inference_steps,
            experiment_config_path,
            data_config,
            alias,
        )
        return self._build_fastapi_code(ctx)

    def _create_generation_context(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]],
        alias: str = "production",
    ) -> GenerationContext:
        """Create generation context from parameters."""
        return GenerationContext(
            pipeline_name=pipeline_name,
            load_map=load_map,
            preprocessor=preprocessor,
            inference_steps=inference_steps,
            experiment_config_path=experiment_config_path,
            data_config=DataConfig.from_dict(data_config),
            model_keys=[inf["model_key"] for inf in inference_steps],
            alias=alias,
        )

    def _build_fastapi_code(self, ctx: GenerationContext) -> str:
        """Build complete FastAPI code."""
        parts = [
            self._gen_header(ctx),
            self._gen_service_init(ctx),
            self._gen_service_methods(),
            self._gen_tabular_inference(ctx.inference_steps),
            self._gen_timeseries_inference(ctx),
            self._gen_service_entrypoint(),
            self._gen_api_init(ctx),
            self._gen_api_endpoints(ctx),
            self._gen_main(),
        ]
        return "".join(parts)

    def _gen_header(self, ctx: GenerationContext) -> str:
        """Generate file header and imports."""
        desc = (
            "Tabular batch prediction"
            if ctx.data_config.data_type == "tabular"
            else "Timeseries multi-step prediction"
        )
        api_type = ctx.data_config.data_type
        features_str = repr(ctx.data_config.features)

        return f"""# Auto-generated FastAPI serve for {ctx.pipeline_name}
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
from mlproject.src.features.facade import FeatureStoreFacade

app = FastAPI(
    title="{ctx.pipeline_name} API",
    version="1.0.0",
    description="Auto-generated serve API for {api_type} data",
)


# Request/Response Schemas

class FeastPredictRequest(BaseModel):
    entities: List[Union[int, str]] = Field(..., description="List of entity IDs")
    entity_key: Optional[str] = Field(None, description="Key to join entities")
    time_point: str = Field(default="now", description="Time point for retrieval")

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
        default={ctx.data_config.output_chunk_length},
        description="Number of steps to predict ahead",
        ge=1
    )


class PredictResponse(BaseModel):
    predictions: Dict[str, List[float]]
    metadata: Optional[Dict[str, Any]] = None

class MultiPredictResponse(BaseModel):
    predictions: Dict[str, Union[List[List[float]], List[float]]]
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_type: str = "{ctx.data_config.data_type}"
    features: List[str] = {features_str}


# Service Implementation

class ServeService:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {ctx.data_config.output_chunk_length}
    FEATURES = {features_str}

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {{}}
        self.feature_store = None

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "{ctx.pipeline_name}")
"""

    def _gen_service_init(self, ctx: GenerationContext) -> str:
        """Generate service initialization code."""
        parts = []

        if ctx.data_config.is_feast:
            parts.append(
                """
            print(f"[ModelService] Initializing Feast Facade...")
            try:
                self.feature_store = FeatureStoreFacade(self.cfg, mode="online")
            except Exception as e:
                print(f"[WARNING] Feast initialization failed: {e}")
                self.feature_store = None
"""
            )

        preprocessor_artifact = self._get_preprocessor_artifact_name(
            ctx.preprocessor, ctx.load_map
        )
        if preprocessor_artifact:
            parts.append(
                f"""
            print(f"[ModelService] Loading preprocessor: {preprocessor_artifact} "
                  f"(alias: {ctx.alias})...")
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{preprocessor_artifact}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.preprocessor = component
"""
            )

        for model_key in set(ctx.model_keys):
            step_id = ctx.load_map.get(model_key, "model")
            parts.append(
                f"""
            print(f"[ModelService] Loading model: {model_key} from {step_id} "
                  f"(alias: {ctx.alias})...")
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{step_id}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.models["{model_key}"] = component
"""
            )

        return "".join(parts)

    def _gen_service_methods(self) -> str:
        """Generate service helper methods."""
        return """
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def get_online_dataset(self,
                           entities: List[Union[int, str]],
                           time_point: str = "now") -> pd.DataFrame:
        if self.feature_store is None:
            raise RuntimeError("Feast not initialized")

        # Use Facade to load features (handles windowing and prefixes)
        print(f"[ModelService] Fetching features for entities: {entities}")
        df = self.feature_store.load_features(
            time_point=time_point,
            entity_ids=entities
        )
        return df

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

    def _gen_tabular_inference(self, inference_steps: List[Dict[str, Any]]) -> str:
        """Generate tabular inference logic."""
        parts = []
        for step_info in inference_steps:
            parts.append(
                f"""
        model = self.models.get("{step_info['model_key']}")
        if model is not None:
            features = context.get("{step_info['features_key']}")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["{step_info['output_key']}"] = preds.tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        key = "{step_info['output_key']}_probabilities"
                        results[key] = proba.tolist()
                    except Exception:
                        pass
"""
            )
        return "".join(parts)

    def _gen_timeseries_inference(self, ctx: GenerationContext) -> str:
        """Generate timeseries inference logic."""
        parts = [
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
        ]

        for step_info in ctx.inference_steps:
            parts.append(
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
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1 and hasattr(current_input, "iloc"):
                        shift = min(self.OUTPUT_CHUNK_LENGTH, len(block_preds))
                        if isinstance(current_input, pd.DataFrame):
                            current_input = current_input.iloc[shift:]
                all_predictions = np.array(all_predictions)
                if all_predictions.ndim == 1:
                    preds_2d = all_predictions
                else:
                    preds_2d = np.concatenate(all_predictions, axis=0)
                results["{step_info['output_key']}"] = preds_2d.tolist()
"""
            )
        return "".join(parts)

    def _gen_service_entrypoint(self) -> str:
        """Generate service entrypoint method."""
        return """
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

    def _gen_api_init(self, ctx: GenerationContext) -> str:
        """Generate API initialization."""
        return f"""
service = ServeService("{ctx.experiment_config_path}")


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


@app.post("/predict", response_model=MultiPredictResponse)
def predict(request: PredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {{"preprocessed_data": preprocessed_data}}
        predictions = service.run_inference_pipeline(context)
        return MultiPredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/predict/feast", response_model=MultiPredictResponse)
def predict_feast(request: FeastPredictRequest) -> MultiPredictResponse:
    try:
        df = service.get_online_dataset(request.entities)
        preprocessed_data = service.preprocess(df)
        context = {{"preprocessed_data": preprocessed_data}}
        predictions = service.run_inference_pipeline(context)
        return MultiPredictResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/feast/batch", response_model=MultiPredictResponse)
def predict_feast_batch(request: FeastPredictRequest) -> MultiPredictResponse:
    return predict_feast(request)

"""

    def _gen_api_endpoints(self, ctx: GenerationContext) -> str:
        """Generate data-type specific endpoints."""
        if ctx.data_config.data_type == "tabular":
            return """
@app.post("/predict/batch", response_model=MultiPredictResponse)
def predict_batch(request: BatchPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
        result = service.predict_tabular_batch(
            context,
            return_probabilities=request.return_probabilities
        )
        return MultiPredictResponse(
            predictions=result["predictions"],
            metadata=result["metadata"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

"""
        return """
@app.post("/predict/multistep", response_model=MultiPredictResponse)
def predict_multistep(request: MultiStepPredictRequest) -> MultiPredictResponse:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < service.INPUT_CHUNK_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Input must have at least "
                       f"{service.INPUT_CHUNK_LENGTH} timesteps (got {len(df)})"
            )
        preprocessed_data = service.preprocess(df)
        context = {"preprocessed_data": preprocessed_data}
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

    def _gen_main(self) -> str:
        """Generate main entry point."""
        return """
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
