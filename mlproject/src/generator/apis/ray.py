from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import ApiGeneratorExtractorsMixin
from .types import DataConfig, GenerationContext


class ApiGeneratorRayServeMixin(ApiGeneratorExtractorsMixin):
    """Ray Serve code generation mixin."""

    def _generate_ray_serve_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate Ray Serve code."""
        ctx = self._create_ray_context(
            pipeline_name,
            load_map,
            preprocessor,
            inference_steps,
            experiment_config_path,
            data_config,
        )
        return self._build_ray_code(ctx)

    def _create_ray_context(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
        data_config: Optional[Dict[str, Any]],
    ) -> GenerationContext:
        """Create generation context for Ray Serve."""
        return GenerationContext(
            pipeline_name=pipeline_name,
            load_map=load_map,
            preprocessor=preprocessor,
            inference_steps=inference_steps,
            experiment_config_path=experiment_config_path,
            data_config=DataConfig.from_dict(data_config),
            model_keys=[inf["model_key"] for inf in inference_steps],
        )

    def _build_ray_code(self, ctx: GenerationContext) -> str:
        """Build complete Ray Serve code."""
        return "".join(
            [
                self._gen_ray_header(ctx),
                self._gen_model_service(ctx),
                self._gen_preprocess_service(ctx),
                self._gen_serve_api(ctx),
                self._gen_ray_main(ctx),
            ]
        )

    def _gen_ray_header(self, ctx: GenerationContext) -> str:
        """Generate Ray Serve header."""
        if ctx.data_config.data_type == "tabular":
            desc = "Tabular batch prediction"
        else:
            desc = "Timeseries multi-step prediction"
        return f"""# Auto-generated Ray Serve deployment for {ctx.pipeline_name}
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
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="{ctx.pipeline_name} Ray Serve API",
    version="1.0.0",
    description="Auto-generated Ray Serve API for {ctx.data_config.data_type} data",
)


# Request/Response Schemas

class PredictRequest(BaseModel):
    data: Dict[str, List[Any]]


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]]
    return_probabilities: bool = False


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]]
    steps_ahead: int = {ctx.data_config.output_chunk_length}


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


# Ray Serve Deployments

@serve.deployment(num_replicas=2, ray_actor_options={{"num_cpus": 0.5}})
class ModelService:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {ctx.data_config.output_chunk_length}
    FEATURES = {repr(ctx.data_config.features)}

    def __init__(self, config_path: str) -> None:
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models: Dict[str, Any] = {{}}
        self.ready = False
        self._load_models()

    def _load_models(self) -> None:
        if not self.mlflow_manager.enabled:
            return
        experiment_name = self.cfg.experiment.get("name", "{ctx.pipeline_name}")
"""

    def _gen_model_service(self, ctx: GenerationContext) -> str:
        """Generate ModelService methods."""
        model_loads = "\n".join(
            [
                f"""        self.models["{key}"] = self.mlflow_manager.load_component(
            name=f"{{experiment_name}}_{ctx.load_map.get(key, 'model')}",
            alias="production",
        )"""
                for key in set(ctx.model_keys)
            ]
        )

        tabular_inference = "\n".join(
            [
                f"""        model = self.models.get("{s['model_key']}")
        if model is not None:
            features = context.get("{s['features_key']}")
            if features is not None:
                x_input = self._prepare_input_tabular(features)
                metadata["n_samples"] = len(x_input)
                preds = model.predict(x_input)
                results["{s['output_key']}"] = preds.tolist()
                if return_probabilities and hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(x_input)
                        results["{s['output_key']}_probabilities"] = proba.tolist()
                    except Exception:
                        pass"""
                for s in ctx.inference_steps
            ]
        )

        ts_inference = "\n".join(
            [
                f"""        model = self.models.get("{s['model_key']}")
        if model is not None:
            features = context.get("{s['features_key']}")
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
                        current_input, "{s['model_type']}"
                    )
                    block_preds = model.predict(x_input)
                    all_predictions.append(block_preds[0])
                    if block_idx < n_blocks - 1:
                        if hasattr(current_input, "iloc"):
                            shift = min(
                                self.OUTPUT_CHUNK_LENGTH,
                                len(block_preds)
                            )
                            if isinstance(current_input, pd.DataFrame):
                                current_input = current_input.iloc[shift:]
                preds_2d = np.concatenate(all_predictions, axis=0)
                results["{s['output_key']}"] = preds_2d.tolist()"""
                for s in ctx.inference_steps
            ]
        )

        return f"""
{model_loads}
        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> None:
        if not self.ready:
            raise RuntimeError("ModelService not ready")

    def _prepare_input_tabular(self, features: Any) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            return features.values
        return np.atleast_2d(np.array(features))

    def _prepare_input_timeseries(
        self, features: Any, model_type: str
    ) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)
        if model_type == "ml":
            return x_input[:self.INPUT_CHUNK_LENGTH, :].reshape(1, -1)
        return x_input[:self.INPUT_CHUNK_LENGTH, :][np.newaxis, :]

    async def predict_tabular_batch(
        self, context: Dict[str, Any], return_probabilities: bool = False
    ) -> Dict[str, Any]:
        results = {{}}
        metadata = {{"n_samples": 0, "model_type": "tabular"}}
{tabular_inference}
        return {{"predictions": results, "metadata": metadata}}

    async def predict_timeseries_multistep(
        self, context: Dict[str, Any], steps_ahead: int
    ) -> Dict[str, Any]:
        results = {{}}
        n_blocks = (
            (steps_ahead + self.OUTPUT_CHUNK_LENGTH - 1) //
            self.OUTPUT_CHUNK_LENGTH
        )
        metadata = {{
            "output_chunk_length": self.OUTPUT_CHUNK_LENGTH,
            "n_blocks": n_blocks,
            "model_type": "timeseries"
        }}
{ts_inference}
        return {{"predictions": results, "metadata": metadata}}

    async def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = await self.predict_tabular_batch(context)
        else:
            result = await self.predict_timeseries_multistep(
                context, steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {{}})

    def is_loaded(self) -> bool:
        return self.ready


"""

    def _gen_preprocess_service(self, ctx: GenerationContext) -> str:
        """Generate PreprocessService."""
        preprocessor_artifact = self._get_preprocessor_artifact_name(
            ctx.preprocessor, ctx.load_map
        )
        prep_load = ""
        if preprocessor_artifact:
            prep_load = (
                f"""        self.preprocessor = """
                f"""self.mlflow_manager.load_component(
            name=f"{{experiment_name}}_{preprocessor_artifact}",
            alias="production",
        )"""
            )

        return f"""@serve.deployment(
    num_replicas=2,
    ray_actor_options={{"num_cpus": 0.5}}
)
class PreprocessService:
    def __init__(self, config_path: str) -> None:
        print("[PreprocessService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.preprocessor = None
        self.ready = False
        self._load_preprocessor()

    def _load_preprocessor(self) -> None:
        if not self.mlflow_manager.enabled:
            self.ready = True
            return
        experiment_name = self.cfg.experiment.get(
            "name", "{ctx.pipeline_name}"
        )
{prep_load}
        self.ready = True
        print("[PreprocessService] Ready")

    def check_health(self) -> None:
        if not self.ready:
            raise RuntimeError("PreprocessService not ready")

    async def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)


"""

    def _gen_serve_api(self, ctx: GenerationContext) -> str:
        """Generate ServeAPI."""
        if ctx.data_config.data_type == "tabular":
            specific_endpoint = """
    @app.post("/predict/batch", response_model=MultiPredictResponse)
    async def predict_batch(
        self, request: BatchPredictRequest
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            result = await self.model_handle.predict_tabular_batch.remote(
                context, request.return_probabilities
            )
            return MultiPredictResponse(
                predictions=result["predictions"],
                metadata=result["metadata"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
"""
        else:
            specific_endpoint = """
    @app.post("/predict/multistep", response_model=MultiPredictResponse)
    async def predict_multistep(
        self, request: MultiStepPredictRequest
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            if len(df) < self.INPUT_CHUNK_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input must have at least "
                           f"{self.INPUT_CHUNK_LENGTH} timesteps"
                )
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {"preprocessed_data": preprocessed_data}
            result = (
                await self.model_handle.predict_timeseries_multistep.remote(
                    context, request.steps_ahead
                )
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

        return f"""@serve.deployment
@serve.ingress(app)
class ServeAPI:
    DATA_TYPE = "{ctx.data_config.data_type}"
    INPUT_CHUNK_LENGTH = {ctx.data_config.input_chunk_length}

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load("{ctx.experiment_config_path}")

    @app.post("/predict", response_model=MultiPredictResponse)
    async def predict(self, request: PredictRequest) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = (
                await self.preprocess_handle.preprocess.remote(df)
            )
            context = {{"preprocessed_data": preprocessed_data}}
            predictions = (
                await self.model_handle.run_inference_pipeline.remote(
                    context
                )
            )
            return MultiPredictResponse(predictions=predictions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
{specific_endpoint}
    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        try:
            model_loaded = await self.model_handle.is_loaded.remote()
        except Exception:
            model_loaded = False
        status = "healthy" if model_loaded else "unhealthy"
        return HealthResponse(
            status=status,
            model_loaded=model_loaded,
            data_type=self.DATA_TYPE
        )


"""

    def _gen_ray_main(self, ctx: GenerationContext) -> str:
        """Generate main function."""
        return f"""def main() -> None:
    ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0")
    serve.start(detached=True)
    config_path = "{ctx.experiment_config_path}"
    model_service = ModelService.bind(config_path)  # type: ignore
    preprocess_service = PreprocessService.bind(
        config_path
    )  # type: ignore
    serve.run(
        ServeAPI.bind(preprocess_service, model_service),  # type: ignore
        route_prefix="/"
    )
    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Press Ctrl+C to stop")
    import signal
    signal.pause()


if __name__ == "__main__":
    main()
"""
