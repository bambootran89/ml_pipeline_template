"""API module for serving time series forecasts.

Provides three endpoints:
- GET /health
- GET /
- POST /predict
"""

import uvicorn
from fastapi import FastAPI, HTTPException

from mlproject.serve.models_service import ModelsService
from mlproject.serve.schemas import PredictRequest

# Initialize FastAPI app
app = FastAPI(title="TS Forecasting API", version="1.0.0")

# Initialize service once at startup
service = ModelsService()


@app.get("/health")
def health_check():
    """
    Return basic health info for the service.

    Includes model load status so external systems
    can verify readiness.
    """
    return {"status": "ok", "model_loaded": service.model is not None}


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
