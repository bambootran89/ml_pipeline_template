"""API code generation mixin for FastAPI and Ray Serve.

This mixin generates API code from serve pipeline configurations,
supporting both FastAPI and Ray Serve frameworks.
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
    """

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Extract inference steps from pipeline.

        Args:
            steps: Pipeline steps list.

        Returns:
            List of inference step configurations.
        """
        inference_steps = []
        for step in steps:
            if step.type == "inference":
                inference_steps.append(self._extract_inference_info(step))
            elif step.type == "branch":
                inference_steps.extend(self._extract_branch_inferences(step))
        return inference_steps

    def _infer_model_type(self, model_key: str) -> str:
        """Infer model type from model key.

        Args:
            model_key: Model key like 'fitted_xgboost_branch'

        Returns:
            'ml' for traditional ML, 'deep_learning' for DL models
        """
        key_lower = model_key.lower()

        # Traditional ML patterns
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

        # Deep learning patterns
        dl_patterns = ["tft", "nlinear", "transformer", "lstm", "gru", "rnn"]

        for pattern in dl_patterns:
            if pattern in key_lower:
                return "deep_learning"

        # Default to ml for unknown
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
        """Get preprocessor artifact name from load_map.

        Args:
            preprocessor: Preprocessor step info
            load_map: Mapping from context_key to step_id

        Returns:
            Artifact name (step_id) or None
        """
        if not preprocessor:
            return None

        instance_key = preprocessor.get("instance_key")
        if not instance_key:
            return None

        # Lookup in load_map to get artifact name (step_id)
        return load_map.get(instance_key)

    def _generate_fastapi_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
    ) -> str:
        """Generate FastAPI code."""
        # Build model keys list
        model_keys = [inf["model_key"] for inf in inference_steps]

        code = f'''"""Auto-generated FastAPI serve for {pipeline_name}.

Generated from serve configuration.
"""

import os
import platform

# Fix for macOS OpenMP library conflicts
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="{pipeline_name} API",
    version="1.0.0",
    description="Auto-generated serve API",
)


class PredictRequest(BaseModel):
    """Prediction request schema."""
    data: Dict[str, List[Any]]


class PredictResponse(BaseModel):
    """Prediction response schema."""
    predictions: Dict[str, List[float]]


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
        self.models = {{}}

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get(
                "name", "{pipeline_name}"
            )

            # Load preprocessor
'''

        # Get preprocessor artifact name from load_map
        preprocessor_artifact = self._get_preprocessor_artifact_name(
            preprocessor, load_map
        )
        if preprocessor_artifact:
            code += f"""            self.preprocessor = (
                self.mlflow_manager.load_component(
                    name=f"{{experiment_name}}_{preprocessor_artifact}",
                    alias="production",
                )
            )
"""

        code += """
            # Load models
"""
        for model_key in set(model_keys):
            # Extract step_id from load_map (artifact name)
            step_id = load_map.get(model_key, "model")
            code += f"""            self.models["{model_key}"] = (
                self.mlflow_manager.load_component(
                    name=f"{{experiment_name}}_{step_id}",
                    alias="production",
                )
            )
"""

        code += '''
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data."""
        if self.preprocessor is None:
            return data
        return self.preprocessor.transform(data)

    def _prepare_input(
        self, features: Any, model_type: str
    ) -> Any:
        """Prepare input shape based on model type.

        Args:
            features: Input features
            model_type: 'ml' or 'deep_learning'

        Returns:
            Reshaped input ready for model
        """
        import numpy as np

        # Get features as numpy array
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)

        # Shape handling based on model type
        if model_type == "ml":
            # Traditional ML: flatten to 2D (n_samples, features)
            if x_input.ndim == 2:
                # Single: (timesteps, features) -> (1, features)
                x_input = x_input.reshape(1, -1)
            elif x_input.ndim == 3:
                # Batch: (n, timesteps, features) -> (n, features)
                x_input = x_input.reshape(x_input.shape[0], -1)
        else:
            # Deep learning: need 3D (n_samples, timesteps, features)
            if x_input.ndim == 2:
                # Single: (timesteps, features) -> (1, timesteps, features)
                x_input = x_input[np.newaxis, :]

        return x_input

    def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Run all inference steps in pipeline.

        Args:
            context: Execution context with features and intermediate results

        Returns:
            Dict mapping output_key to predictions for each inference step
        """
        results = {}

'''

        # Generate inference code for each step
        for step_info in inference_steps:
            step_id = step_info["id"]
            model_key = step_info["model_key"]
            features_key = step_info["features_key"]
            output_key = step_info["output_key"]
            model_type = step_info["model_type"]

            code += f"""        # Step: {step_id} (model_type: {model_type})
        model = self.models.get("{model_key}")
        if model is not None:
            features = context.get("{features_key}")
            if features is not None:
                x_input = self._prepare_input(features, "{model_type}")
                preds = model.predict(x_input)
                results["{output_key}"] = preds.flatten().tolist()
                context["{output_key}"] = preds

"""

        code += """        return results


# Initialize service
"""

        code += f'''service = ServeService("{experiment_config_path}")


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
        preprocessed_data = service.preprocess(df)

        # Build context with preprocessed data
        context = {{"preprocessed_data": preprocessed_data}}

        # Run all inference steps in pipeline
        predictions = service.run_inference_pipeline(context)

        return PredictResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        return code

    def _generate_ray_serve_code(
        self,
        pipeline_name: str,
        load_map: Dict[str, str],
        preprocessor: Optional[Dict[str, Any]],
        inference_steps: List[Dict[str, Any]],
        experiment_config_path: str,
    ) -> str:
        """Generate Ray Serve code."""
        model_keys = [inf["model_key"] for inf in inference_steps]

        code = f'''"""Auto-generated Ray Serve deployment for {pipeline_name}.

Generated from serve configuration.
"""

import os
import platform

# Fix for macOS OpenMP library conflicts
if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve

from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.utils.config_class import ConfigLoader

app = FastAPI(
    title="{pipeline_name} Ray Serve API",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Prediction request."""
    data: Dict[str, List[Any]]


class PredictResponse(BaseModel):
    """Prediction response."""
    predictions: Dict[str, List[float]]


class HealthResponse(BaseModel):
    """Health response."""
    status: str
    model_loaded: bool


@serve.deployment(num_replicas=2, ray_actor_options={{"num_cpus": 0.5}})
class ModelService:
    """Model inference service."""

    def __init__(self, config_path: str) -> None:
        """Initialize model service."""
        print("[ModelService] Initializing...")
        self.cfg = ConfigLoader.load(config_path)
        self.mlflow_manager = MLflowManager(self.cfg)
        self.models: Dict[str, Any] = {{}}
        self.ready = False

        self._load_models()

    def _load_models(self) -> None:
        """Load models from MLflow."""
        if not self.mlflow_manager.enabled:
            return

        experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")

        # Load models
'''
        for model_key in set(model_keys):
            step_id = load_map.get(model_key, "model")
            code += f"""        self.models["{model_key}"] = (
            self.mlflow_manager.load_component(
                name=f"{{experiment_name}}_{step_id}",
                alias="production",
            )
        )
"""

        code += '''
        self.ready = True
        print("[ModelService] Ready")

    def check_health(self) -> None:
        """Health check."""
        if not self.ready:
            raise RuntimeError("ModelService not ready")

    def _prepare_input(
        self, features: Any, model_type: str
    ) -> Any:
        """Prepare input shape based on model type.

        Args:
            features: Input features
            model_type: 'ml' or 'deep_learning'

        Returns:
            Reshaped input ready for model
        """
        # Get features as numpy array
        if isinstance(features, pd.DataFrame):
            x_input = features.values
        else:
            x_input = np.array(features)

        # Shape handling based on model type
        if model_type == "ml":
            # Traditional ML: flatten to 2D (n_samples, features)
            if x_input.ndim == 2:
                # Single: (timesteps, features) -> (1, features)
                x_input = x_input.reshape(1, -1)
            elif x_input.ndim == 3:
                # Batch: (n, timesteps, features) -> (n, features)
                x_input = x_input.reshape(x_input.shape[0], -1)
        else:
            # Deep learning: need 3D
            if x_input.ndim == 2:
                # Single: (timesteps, features) -> (1, timesteps, features)
                x_input = x_input[np.newaxis, :]

        return x_input

    async def run_inference_pipeline(
        self, context: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Run all inference steps in pipeline.

        Args:
            context: Execution context with features and intermediate results

        Returns:
            Dict mapping output_key to predictions for each inference step
        """
        results = {}

'''

        # Generate inference code for each step (same as FastAPI)
        for step_info in inference_steps:
            step_id = step_info["id"]
            model_key = step_info["model_key"]
            features_key = step_info["features_key"]
            output_key = step_info["output_key"]
            model_type = step_info["model_type"]

            code += f"""        # Step: {step_id} (model_type: {model_type})
        model = self.models.get("{model_key}")
        if model is not None:
            features = context.get("{features_key}")
            if features is not None:
                x_input = self._prepare_input(features, "{model_type}")
                preds = model.predict(x_input)
                results["{output_key}"] = preds.flatten().tolist()
                context["{output_key}"] = preds

"""

        code += '''        return results

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
'''

        preprocessor_artifact = self._get_preprocessor_artifact_name(
            preprocessor, load_map
        )
        if preprocessor_artifact:
            code += f"""        self.preprocessor = self.mlflow_manager.load_component(
            name=f"{{experiment_name}}_{preprocessor_artifact}",
            alias="production",
        )
"""

        code += '''
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
        self.cfg = ConfigLoader.load(""'''

        code += f'''"{experiment_config_path}"'''

        code += '''"")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Prediction endpoint."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(request.data)

            # Preprocess
            preprocessed_data = await self.preprocess_handle.preprocess.remote(df)

            # Build context with preprocessed data
            context = {"preprocessed_data": preprocessed_data}

            # Run all inference steps in pipeline
            predictions = await self.model_handle.run_inference_pipeline.remote(context)

            return PredictResponse(predictions=predictions)

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

'''
        code += f"""    # Bind services
    config_path = "{experiment_config_path}"
    # type: ignore comments for Ray Serve .bind() dynamic attributes
    model_service = ModelService.bind(config_path)  # type: ignore
    preprocess_service = PreprocessService.bind(config_path)  # type: ignore

    # Deploy API
    serve.run(
        ServeAPI.bind(preprocess_service, model_service),  # type: ignore
        route_prefix="/",
    )

    print("[Ray Serve] API ready at http://localhost:8000")
    print("[Ray Serve] Press Ctrl+C to stop")

    import signal
    signal.pause()


if __name__ == "__main__":
    main()
"""

        return code

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
    ) -> str:
        """Generate API code from serve configuration.

        Args:
            serve_config_path: Path to serve YAML config.
            output_dir: Directory to write generated API code.
            framework: Either 'fastapi' or 'ray' for Ray Serve.
            experiment_config_path: Path to experiment config.

        Returns:
            Path to generated API file.

        Raises:
            ValueError: If framework is not supported.
        """
        # Load serve config
        cfg = OmegaConf.load(serve_config_path)
        assert isinstance(cfg, DictConfig)

        pipeline_name = cfg.pipeline.name
        steps = cfg.pipeline.steps

        # Extract components
        load_map = self._extract_load_map(steps)
        preprocessor = self._extract_preprocessor_info(steps)
        inference_steps = self._extract_inference_steps(steps)

        # Generate code based on framework
        if framework == "fastapi":
            code = self._generate_fastapi_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
            )
            filename = f"{pipeline_name}_fastapi.py"
        elif framework == "ray":
            code = self._generate_ray_serve_code(
                pipeline_name,
                load_map,
                preprocessor,
                inference_steps,
                experiment_config_path,
            )
            filename = f"{pipeline_name}_ray.py"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        # Write to file
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[ApiGenerator] Generated {framework} API: {output_path}")
        return str(output_path)
