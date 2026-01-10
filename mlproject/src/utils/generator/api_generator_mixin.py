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

    def _extract_inference_info(self, step: Any) -> Dict[str, Any]:
        """Extract info from single inference step."""
        return {
            "id": step.id,
            "model_key": step.wiring.inputs.model,
            "features_key": step.wiring.inputs.features,
            "output_key": step.wiring.outputs.predictions,
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
                    "instance_key": step.instance_key,
                }
        return None

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
        primary_model = model_keys[0] if model_keys else "model"

        code = f'''"""Auto-generated FastAPI serve for {pipeline_name}.

Generated from serve configuration.
"""

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
        self.models = {{}}

        if self.mlflow_manager.enabled:
            experiment_name = self.cfg.experiment.get("name", "{pipeline_name}")

            # Load preprocessor
'''

        if preprocessor:
            code += """            self.preprocessor = (
                self.mlflow_manager.load_component(
                    name=f"{experiment_name}_preprocessor",
                    alias="production",
                )
            )
"""

        code += """
            # Load models
"""
        for model_key in set(model_keys):
            # Extract step_id from load_map
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

    def _get_input_chunk_length(self) -> int:
        """Get input chunk length from config."""
        if hasattr(self.cfg, "experiment") and hasattr(
            self.cfg.experiment, "hyperparams"
        ):
            hyperparams = self.cfg.experiment.hyperparams
            return int(hyperparams.get("input_chunk_length", 24))
        return 24

    def predict(self, features: pd.DataFrame, model_key: str) -> List[float]:
        """Run model inference."""
        model = self.models.get(model_key)
        if model is None:
            raise RuntimeError(f"Model {model_key} not loaded")

        # Prepare input
        input_length = self._get_input_chunk_length()
        x_input = features.values[-input_length:]
        import numpy as np
        x_input = x_input[np.newaxis, :].astype(np.float32)

        # Predict
        preds = model.predict(x_input)
        return preds.flatten().tolist()


# Initialize service
'''

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
        features = service.preprocess(df)

        # Predict with primary model
        predictions = service.predict(features, "{primary_model}")

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
        primary_model = model_keys[0] if model_keys else "model"

        code = f'''"""Auto-generated Ray Serve deployment for {pipeline_name}.

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
    title="{pipeline_name} Ray Serve API",
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
'''

        if preprocessor:
            code += """        self.preprocessor = self.mlflow_manager.load_component(
            name=f"{experiment_name}_preprocessor",
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
        self.cfg = ConfigLoader.load("'''

        code += f'''"{experiment_config_path}"'''

        code += '''")
        self.input_chunk_length = self._get_input_chunk_length()

    def _get_input_chunk_length(self) -> int:
        """Get input chunk length from config."""
        if hasattr(self.cfg, "experiment") and hasattr(
            self.cfg.experiment, "hyperparams"
        ):
            hyperparams = self.cfg.experiment.hyperparams
            return int(hyperparams.get("input_chunk_length", 24))
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
'''
        code += f"""            preds = await self.model_handle.predict.remote(
                x_input, "{primary_model}"
            )
"""

        code += '''
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

'''
        code += f"""    # Bind services
    config_path = "{experiment_config_path}"
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
