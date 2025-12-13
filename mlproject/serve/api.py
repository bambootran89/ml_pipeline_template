"""API module for serving time series forecasts.

Provides three endpoints:
- GET /health
- GET /
- POST /predict
"""

import os
from datetime import datetime

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mlproject.serve.models_service import ModelsService
from mlproject.serve.schemas import PredictRequest

# Initialize FastAPI app
app = FastAPI(title="TS Forecasting API", version="1.0.0")

# Initialize service once at startup
service = ModelsService()


def _check_mlflow_connection() -> bool:
    """Ping MLflow server"""
    try:
        response = requests.get(f"{os.getenv('MLFLOW_TRACKING_URI')}/health", timeout=2)
        return response.status_code == 200
    except BaseException:
        return False


class HealthResponse(BaseModel):
    """
    Response schema for the health check endpoint.

    This model represents the runtime health status of the service,
    including system information and model readiness indicators.
    It is typically returned by the `/health` or `/healthz` endpoint
    for monitoring, observability, and orchestration (e.g. Kubernetes).

    Attributes:
        status (str):
            Overall service status (e.g. "ok", "degraded", "unhealthy").

        timestamp (datetime):
            UTC timestamp when the health check was generated.

        model_loaded (bool):
            Indicates whether the ML model has been successfully loaded
            and is ready to serve inference requests.

        memory_usage_mb (float):
            Current memory usage of the service process in megabytes.

        dependencies (dict):
            Health status of external dependencies (e.g. database,
            message broker, feature store), typically represented as
            key–value pairs where values indicate availability or latency.
    """
    status: str
    timestamp: datetime
    model_loaded: bool
    memory_usage_mb: float
    dependencies: dict


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    K8s liveness/readiness probe endpoint

    Returns 200 if:
    - Model loaded
    - Memory < 90% threshold
    - MLflow connection alive
    """
    checks = {
        "model_loaded": service.model is not None,
        "mlflow_reachable": _check_mlflow_connection(),
        "disk_space_ok": psutil.disk_usage("/").percent < 90,
    }

    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

    all_healthy = all(checks.values()) and memory_mb < 2000  # 2GB limit

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=checks["model_loaded"],
        memory_usage_mb=round(memory_mb, 2),
        dependencies=checks,
    )


@app.get("/")
def root():
    """
    Root endpoint with a simple status message.

    Useful for quick checks that the API is online.
    """
    return {"message": "TimeSeries Forecasting API is running"}


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Run prediction using the loaded model.

    Expects a PredictRequest containing historical
    data. Returns model output in JSON format.

    Raises:
        HTTPException: When input is invalid (400) or
        when an unexpected error occurs (500).
    """
    try:
        result = service.predict(request)
        return result

    except ValueError as e:
        # explicitly chain exception
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        # explicitly chain exception
        raise HTTPException(
            status_code=500, detail=f"Internal Server Error: {str(e)}"
        ) from e


if __name__ == "__main__":
    uvicorn.run("mlproject.serve.api:app", host="0.0.0.0", port=8000, reload=True)
