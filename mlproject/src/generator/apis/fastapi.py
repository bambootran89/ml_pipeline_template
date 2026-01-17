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

    def _get_feature_generator_keys(self, ctx: GenerationContext) -> List[str]:
        """Get list of feature generator model keys."""
        return [fg.model_key for fg in ctx.data_config.feature_generators]

    def _gen_feature_generators_config(self, ctx: GenerationContext) -> str:
        """Generate feature generators configuration as Python dict literal."""
        if not ctx.data_config.feature_generators:
            return "{}"

        items = []
        for fg in ctx.data_config.feature_generators:
            items.append(
                f'        "{fg.output_key}": {{'
                f'"model_key": "{fg.model_key}", '
                f'"artifact_name": "{fg.artifact_name}", '
                f'"inference_method": "{fg.inference_method}", '
                f'"step_type": "{fg.step_type}"}}'
            )

        return "{\n" + ",\n".join(items) + "\n    }"

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
    ADDITIONAL_FEATURE_KEYS = {repr(ctx.data_config.additional_feature_keys)}
    FEATURE_GENERATORS = {self._gen_feature_generators_config(ctx)}

    def __init__(self, config_path: str) -> None:
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.models = {{}}
        self.feature_generators = {{}}
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

        # Load feature generators from sub-pipeline
        for fg in ctx.data_config.feature_generators:
            parts.append(
                f"""
            # Load feature generator: {fg.step_id}
            print(f"[ModelService] Loading feature generator: {fg.artifact_name} "
                  f"(alias: {ctx.alias})...")
            component = self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{fg.artifact_name}",
                alias="{ctx.alias}",
            )
            if component is not None:
                self.feature_generators["{fg.output_key}"] = {{
                    "model": component,
                    "method": "{fg.inference_method}",
                    "type": "{fg.step_type}",
                }}
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

    def generate_additional_features(self, base_features: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Generate additional features using feature generators (sub-pipeline models).

        Runs each feature generator model on the base features and returns
        a dict of output_key -> generated_features.

        Args:
            base_features: Preprocessed base features DataFrame.

        Returns:
            Dict mapping output_key to generated feature arrays.
        \"\"\"
        additional_features = {}

        if not self.feature_generators:
            return additional_features

        print(f"[ModelService] Generating additional features from "
              f"{len(self.feature_generators)} generators...")

        # Prepare input
        x_input = base_features.values if isinstance(base_features, pd.DataFrame) else base_features

        for output_key, fg_info in self.feature_generators.items():
            model = fg_info["model"]
            method = fg_info["method"]
            fg_type = fg_info["type"]

            try:
                # Get inference method
                inference_fn = getattr(model, method, None)
                if inference_fn is None:
                    # Fallback to transform or predict
                    inference_fn = getattr(model, "transform", None) or getattr(model, "predict", None)

                if inference_fn is None:
                    print(f"  Warning: {output_key} has no {method}/transform/predict method, skipping")
                    continue

                # Run inference
                result = inference_fn(x_input)

                # Store result
                additional_features[output_key] = result
                result_shape = result.shape if hasattr(result, "shape") else len(result)
                print(f"  + {output_key} ({fg_type}): {result_shape}")

            except Exception as e:
                print(f"  Warning: Failed to generate {output_key}: {e}")
                continue

        return additional_features

    def compose_features(
        self,
        base_features: pd.DataFrame,
        additional_features: Dict[str, Any]
    ) -> pd.DataFrame:
        \"\"\"Compose base features with additional generated features.

        Args:
            base_features: Base preprocessed features.
            additional_features: Dict of output_key -> generated features.

        Returns:
            Composed DataFrame with all features.
        \"\"\"
        if not additional_features:
            return base_features

        composed = base_features.copy() if isinstance(base_features, pd.DataFrame) else pd.DataFrame(base_features)
        n_samples = len(composed)

        print(f"[ModelService] Composing features: base {composed.shape}")

        for key, features in additional_features.items():
            # Convert to DataFrame
            if isinstance(features, np.ndarray):
                if features.ndim == 1:
                    feat_df = pd.DataFrame({f"{key}_0": features})
                else:
                    cols = [f"{key}_{i}" for i in range(features.shape[1])]
                    feat_df = pd.DataFrame(features, columns=cols)
            elif isinstance(features, pd.DataFrame):
                feat_df = features.copy()
                feat_df.columns = [f"{key}_{c}" for c in feat_df.columns]
            else:
                feat_df = pd.DataFrame({f"{key}_0": features})

            # Align lengths
            if len(feat_df) != n_samples:
                if len(feat_df) == 1:
                    # Broadcast
                    feat_df = pd.concat([feat_df] * n_samples, ignore_index=True)
                elif len(feat_df) > n_samples:
                    # Truncate
                    feat_df = feat_df.iloc[:n_samples]
                else:
                    # Pad at start
                    n_pad = n_samples - len(feat_df)
                    pad_df = pd.concat([feat_df.iloc[[0]]] * n_pad, ignore_index=True)
                    feat_df = pd.concat([pad_df, feat_df], ignore_index=True)

            # Reset index for alignment
            feat_df.index = composed.index

            # Concatenate
            composed = pd.concat([composed, feat_df], axis=1)
            print(f"  + {key}: {feat_df.shape} -> Total: {composed.shape}")

        return composed

    def run_full_pipeline(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Run full inference pipeline including feature generation.

        This is the main entry point that:
        1. Preprocesses raw data
        2. Generates additional features via feature generators
        3. Composes all features
        4. Runs model predictions

        Args:
            raw_data: Raw input DataFrame.

        Returns:
            Dict with predictions and metadata.
        \"\"\"
        # Step 1: Preprocess
        preprocessed = self.preprocess(raw_data)

        # Step 2: Generate additional features (if any feature generators exist)
        additional_features = self.generate_additional_features(preprocessed)

        # Step 3: Compose features
        composed = self.compose_features(preprocessed, additional_features)

        # Step 4: Run predictions
        context = {"preprocessed_data": composed}

        if self.DATA_TYPE == "tabular":
            result = self.predict_tabular_batch(context)
        else:
            result = self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )

        return result

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
        # Use full pipeline with feature generation and composition
        result = service.run_full_pipeline(df)
        return MultiPredictResponse(
            predictions=result.get("predictions", {{}}),
            metadata=result.get("metadata")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/predict/feast", response_model=MultiPredictResponse)
def predict_feast(request: FeastPredictRequest) -> MultiPredictResponse:
    try:
        df = service.get_online_dataset(request.entities)
        # Use full pipeline with feature generation and composition
        result = service.run_full_pipeline(df)
        return MultiPredictResponse(
            predictions=result.get("predictions", {{}}),
            metadata=result.get("metadata")
        )
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
        # Use full pipeline with feature generation and composition
        result = service.run_full_pipeline(df)
        # Add probabilities if requested
        if request.return_probabilities:
            for key, model in service.models.items():
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(df.values)
                        result["predictions"][f"{key}_probabilities"] = proba.tolist()
                    except Exception:
                        pass
        return MultiPredictResponse(
            predictions=result.get("predictions", {}),
            metadata=result.get("metadata")
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
        # Use full pipeline with feature generation and composition
        result = service.run_full_pipeline(df)
        return MultiPredictResponse(
            predictions=result.get("predictions", {}),
            metadata=result.get("metadata")
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
