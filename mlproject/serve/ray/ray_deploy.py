"""
Ray Serve deployment for time-series forecasting.

Run:
    python mlproject/serve/ray/ray_deploy.py
"""

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ray
from fastapi import FastAPI, HTTPException
from ray import serve

from mlproject.serve.schemas import PredictRequest
from mlproject.src.models.nlinear_wrapper import NLinearWrapper
from mlproject.src.models.tft_wrapper import TFTWrapper
from mlproject.src.pipeline.config_loader import ConfigLoader
from mlproject.src.preprocess.online import serve_preprocess_request

# --------------------------
# Config / Artifacts paths
# --------------------------
ARTIFACTS_DIR = os.path.join("mlproject", "artifacts", "models")
CONFIG_PATH = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="mlproject Ray Serve Forecast API")


# --------------------------
# Ray Deployments
# --------------------------
@serve.deployment  # type: ignore
class PreprocessingService:
    """Ray deployment for preprocessing input data."""

    def __init__(self):
        """Initialize preprocessing service with config."""
        self.cfg = ConfigLoader.load(CONFIG_PATH)

    def preprocess(self, data_dict: Dict) -> pd.DataFrame:
        """
        Preprocess raw input dictionary into a transformed DataFrame.

        Args:
            data_dict: Raw input data {"feature": [values...]}

        Returns:
            pd.DataFrame: Preprocessed features
        """
        df = pd.DataFrame(data_dict)
        if "date" in df.columns:
            df = df.set_index("date")
        return serve_preprocess_request(df, self.cfg)


@serve.deployment  # type: ignore
class ModelService:
    """Ray deployment for model inference."""

    def __init__(self):
        """Initialize model service and load the trained model."""
        cfg = ConfigLoader.load(CONFIG_PATH)
        self.cfg = cfg
        experiment = self.cfg.experiment
        self.input_chunk_length = experiment.get("hyperparams", {}).get(
            "input_chunk_length", 24
        )
        model_name = experiment.get("model")
        hp = experiment.get("hyperparams", {})

        if model_name == "nlinear":
            self.model = NLinearWrapper(hp)
        elif model_name == "tft":
            self.model = TFTWrapper(hp)
        else:
            raise RuntimeError(f"Unknown model {model_name}")

        self.model.load(ARTIFACTS_DIR)
        self.model_loaded = True

    def predict(self, x_input: np.ndarray) -> List[float]:
        """
        Make predictions from input window.

        Args:
            x_input: Input array [1, seq_len, n_features]

        Returns:
            List[float]: Predicted values
        """
        preds = self.model.predict(x_input)
        return preds.flatten().tolist()

    def is_loaded(self) -> bool:
        """Check if the model has been loaded successfully."""
        return self.model_loaded


@serve.deployment  # type: ignore
@serve.ingress(app)  # type: ignore
class ForecastAPI:
    """Main API deployment handling HTTP requests."""

    def __init__(self, preprocess_handle: Any, model_handle: Any):
        """
        Initialize API with service handles.

        Args:
            preprocess_handle: Handle to PreprocessingService
            model_handle: Handle to ModelService
        """
        self.preprocess_handle = preprocess_handle
        self.model_handle = model_handle

    @app.post("/predict")
    async def predict(self, req: PredictRequest):
        """
        Prediction endpoint.

        Args:
            req: PredictRequest containing input data

        Returns:
            dict: {"prediction": [values...]}

        Raises:
            HTTPException: 400 if input is invalid, 500 if prediction fails
        """
        try:
            # Step 1: Preprocess input
            df_transformed = await self.preprocess_handle.preprocess.remote(req.data)

            # Step 2: Check input length
            input_chunk_length = self.model_handle.input_chunk_length
            if len(df_transformed) < input_chunk_length:
                raise ValueError(
                    f"Input data has {len(df_transformed)} rows, \
                        need at least {input_chunk_length}"
                )

            # Step 3: Prepare input window
            x_input = df_transformed.iloc[-input_chunk_length:].values[np.newaxis, :]

            # Step 4: Run prediction
            preds = await self.model_handle.predict.remote(x_input)
            return {"prediction": preds}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {e}"
            ) from e

    @app.get("/health")
    async def health_check(self):
        """
        Health check endpoint.

        Returns:
            dict: {"status": "ok", "model_loaded": bool}
        """
        try:
            model_loaded = await self.model_handle.is_loaded.remote()
        except Exception:
            model_loaded = False
        return {"status": "ok", "model_loaded": model_loaded}


def main():
    """
    Main entry point to deploy the Ray Serve application.

    Steps:
        1. Initialize Ray
        2. Start Ray Serve
        3. Bind deployment handles
        4. Deploy ForecastAPI
    """
    # Start Ray + Serve
    ray.init(ignore_reinit_error=True)
    serve.start(detached=True)

    # Bind handles (pylint ignore dynamic attributes)
    preprocess_handle = PreprocessingService.bind()  # pylint: disable=no-member
    model_handle = ModelService.bind()  # pylint: disable=no-member

    # Deploy the API
    serve.run(
        ForecastAPI.bind(preprocess_handle, model_handle),  # pylint: disable=no-member
        route_prefix="/",
    )

    print("[Ray Serve] Deployment successful!")
    print("[Ray Serve] API available at http://localhost:8000")
    print("[Ray Serve] Swagger docs at http://localhost:8000/docs")


if __name__ == "__main__":
    main()
