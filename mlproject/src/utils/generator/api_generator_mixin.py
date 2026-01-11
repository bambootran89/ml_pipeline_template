"""API code generation mixin for FastAPI and Ray Serve.

This mixin generates API code from serve pipeline configurations,
supporting both FastAPI and Ray Serve frameworks.

Supports:
- Tabular data: batch prediction for multiple rows
- Timeseries data: multi-step prediction with configurable horizon
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


class ApiGeneratorMixin:
    """Mixin for generating API code from serve configurations.

    Provides methods to generate FastAPI and Ray Serve code from
    serve pipeline YAML configs. Uses template-based generation
    to keep complexity low.

    Supports both tabular and timeseries data types with appropriate
    prediction strategies (batch for tabular, multi-step for timeseries).
    """

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Extract inference steps from pipeline."""
        inference_steps = []
        for step in steps:
            if step.type == "inference":
                inference_steps.append(self._extract_inference_info(step))
            elif step.type == "branch":
                inference_steps.extend(self._extract_branch_inferences(step))
        return inference_steps

    def _infer_model_type(self, model_key: str) -> str:
        """Infer model type from model key."""
        key_lower = model_key.lower()

        ml_patterns = [
            "xgboost",
            "xgb",
            "catboost",
            "kmeans",
            "kmean",
            "lightgbm",
            "lgbm",
            "randomforest",
            "rf",
        ]
        for pattern in ml_patterns:
            if pattern in key_lower:
                return "ml"

        dl_patterns = ["tft", "nlinear", "transformer", "lstm", "gru", "rnn"]
        for pattern in dl_patterns:
            if pattern in key_lower:
                return "deep_learning"

        return "ml"

    def _extract_inference_info(self, step: Any) -> Dict[str, Any]:
        """Extract info from single inference step."""
        model_key = step.wiring.inputs.model
        return {
            "id": step.id,
            "model_key": model_key,
            "features_key": step.wiring.inputs.features,
            "output_key": step.wiring.outputs.predictions,
            "model_type": self._infer_model_type(model_key),
        }

    def _extract_branch_inferences(self, branch_step: Any) -> List[Dict[str, Any]]:
        """Extract inference info from branch step."""
        inferences = []
        for branch_name in ["if_true", "if_false"]:
            if hasattr(branch_step, branch_name):
                branch = getattr(branch_step, branch_name)
                if branch.type == "inference":
                    inferences.append(self._extract_inference_info(branch))
        return inferences

    def _extract_load_map(self, steps: List[Any]) -> Dict[str, str]:
        """Extract model loading configuration."""
        load_map = {}
        for step in steps:
            if step.type == "mlflow_loader":
                for item in step.load_map:
                    load_map[item.context_key] = item.step_id
        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract preprocessor configuration."""
        for step in steps:
            if step.type == "preprocessor":
                return {
                    "id": step.id,
                    "instance_key": getattr(step, "instance_key", None),
                }
        return None

    def _get_preprocessor_artifact_name(
        self, preprocessor: Optional[Dict[str, Any]], load_map: Dict[str, str]
    ) -> Optional[str]:
        """Get preprocessor artifact name from load_map."""
        if not preprocessor:
            return None
        instance_key = preprocessor.get("instance_key")
        if not instance_key:
            return None
        return load_map.get(instance_key)

    def _extract_data_config(self, cfg: DictConfig) -> Dict[str, Any]:
        """Extract data configuration from config."""
        data_config = {
            "data_type": "timeseries",
            "features": [],
            "target_columns": [],
            "input_chunk_length": 24,
            "output_chunk_length": 6,
        }

        if hasattr(cfg, "data"):
            data = cfg.data
            data_config["data_type"] = getattr(data, "type", "timeseries")
            data_config["features"] = list(getattr(data, "features", []))
            data_config["target_columns"] = list(getattr(data, "target_columns", []))

        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "hyperparams"):
            hp = cfg.experiment.hyperparams
            data_config["input_chunk_length"] = getattr(hp, "input_chunk_length", 24)
            data_config["output_chunk_length"] = getattr(hp, "output_chunk_length", 6)

        return data_config

    def _generate_fastapi_code(
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

        code_parts = []

        desc = (
            "Tabular batch prediction"
            if data_type == "tabular"
            else "Timeseries multi-step prediction"
        )
        api_desc = "tabular" if data_type == "tabular" else "timeseries"

        code_parts.append(
            f"""# Auto-generated Ray Serve deployment for {pipeline_name}
# Generated from serve configuration.
# Supports: {desc}

import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ray import serve

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="{pipeline_name} Ray Serve API",
    version="1.0.0",
    description="Auto-generated Ray Serve API for {api_desc} data",
)


# Request/Response Schemas

class PredictRequest(BaseModel):
    data: Dict[str, List[Any]]


class BatchPredictRequest(BaseModel):
    data: Dict[str, List[Any]]
    return_probabilities: bool = False


class MultiStepPredictRequest(BaseModel):
    data: Dict[str, List[Any]]
    steps_ahead: int = {output_chunk_length}


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


# Ray Serve Deployments

@serve.deployment(num_replicas=2, ray_actor_options={{"num_cpus": 0.5}})
class ModelService:
    DATA_TYPE = "{data_type}"
    INPUT_CHUNK_LENGTH = {input_chunk_length}
    OUTPUT_CHUNK_LENGTH = {output_chunk_length}
    FEATURES = {features}

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
        experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")
"""
        )

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

        code_parts.append(
            """
        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> None:
        if not self.ready:
            raise RuntimeError("ModelService not ready")

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

    async def predict_tabular_batch(
        self,
        context: Dict[str, Any],
        return_probabilities: bool = False
    ) -> Dict[str, Any]:
        results = {}
        metadata = {"n_samples": 0, "model_type": "tabular"}
"""
        )

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

    async def predict_timeseries_multistep(
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

    async def run_inference_pipeline(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        if self.DATA_TYPE == "tabular":
            result = await self.predict_tabular_batch(context)
        else:
            result = await self.predict_timeseries_multistep(
                context,
                steps_ahead=self.OUTPUT_CHUNK_LENGTH
            )
        return result.get("predictions", {})

    def is_loaded(self) -> bool:
        return self.ready


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5})
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
"""
        )

        code_parts.append(
            f"""
        experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")
"""
        )

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

        code_parts.append(
            """
        self.ready = True
        print("[PreprocessService] Ready")

    def check_health(self) -> None:
        if not self.ready:
            raise RuntimeError("PreprocessService not ready")

    async def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)


@serve.deployment
@serve.ingress(app)
class ServeAPI:
"""
        )

        code_parts.append(
            f"""
    DATA_TYPE = "{data_type}"
    INPUT_CHUNK_LENGTH = {input_chunk_length}

    def __init__(self, preprocess_handle: Any, model_handle: Any) -> None:
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle
        self.cfg = ConfigLoader.load("{experiment_config_path}")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = await self.preprocess_handle.preprocess.remote(df)
            context = {{"preprocessed_data": preprocessed_data}}
            predictions = await self.model_handle.run_inference_pipeline.remote(context)
            return PredictResponse(predictions=predictions)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

"""
        )

        if data_type == "tabular":
            code_parts.append(
                """
    @app.post("/predict/batch", response_model=PredictResponse)
    async def predict_batch(self, request: BatchPredictRequest) -> PredictResponse:
        try:
            df = pd.DataFrame(request.data)
            preprocessed_data = await self.preprocess_handle.preprocess.remote(df)
            context = {"preprocessed_data": preprocessed_data}
            result = await self.model_handle.predict_tabular_batch.remote(
                context,
                request.return_probabilities
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
    async def predict_multistep(
        self,
        request: MultiStepPredictRequest
    ) -> MultiPredictResponse:
        try:
            df = pd.DataFrame(request.data)
            if len(df) < self.INPUT_CHUNK_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input must have at \
                      {self.INPUT_CHUNK_LENGTH} timesteps"
                )
            preprocessed_data = await self.preprocess_handle.preprocess.remote(df)
            context = {"preprocessed_data": preprocessed_data}
            result = await self.model_handle.predict_timeseries_multistep.remote(
                context,
                request.steps_ahead
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
            f"""
    @app.get("/health", response_model=HealthResponse)
    async def health(self) -> HealthResponse:
        try:
            model_loaded = await self.model_handle.is_loaded.remote()
        except Exception:
            model_loaded = False
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            data_type=self.DATA_TYPE
        )


def main() -> None:
    ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0")
    serve.start(detached=True)

    config_path = "{experiment_config_path}"
    model_service = ModelService.bind(config_path)  # type: ignore
    preprocess_service = PreprocessService.bind(config_path)  # type: ignore

    serve.run(
        ServeAPI.bind( # type: ignore
            preprocess_service,
            model_service
        ),
        route_prefix="/"
    )

    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Press Ctrl+C to stop")

    import signal
    signal.pause()


if __name__ == "__main__":
    main()
"""
        )

        return "".join(code_parts)

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
    ) -> str:
        """Generate API code from serve configuration."""
        cfg = OmegaConf.load(serve_config_path)
        assert isinstance(cfg, DictConfig)

        pipeline_name = cfg.pipeline.name
        steps = cfg.pipeline.steps

        load_map = self._extract_load_map(steps)
        preprocessor = self._extract_preprocessor_info(steps)
        inference_steps = self._extract_inference_steps(steps)

        # Extract data config from experiment config (preferred) or serve config
        if experiment_config_path and Path(experiment_config_path).exists():
            exp_cfg = OmegaConf.load(experiment_config_path)
            data_config = self._extract_data_config(exp_cfg)
        else:
            data_config = self._extract_data_config(cfg)

        if framework == "fastapi":
            code = self._generate_fastapi_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
                data_config,
            )
            filename = f"{pipeline_name}_fastapi.py"
        elif framework == "ray":
            code = self._generate_ray_serve_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
                data_config,
            )
            filename = f"{pipeline_name}_ray.py"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[ApiGenerator] Generated {framework} API: {output_path}")
        print(f"[ApiGenerator] Data type: {data_config.get('data_type', 'unknown')}")
        return str(output_path)
