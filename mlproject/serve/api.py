from fastapi import FastAPI, HTTPException

from mlproject.serve.models_service import ModelService
from mlproject.serve.schemas import PredictRequest

app = FastAPI(title="mlproject Forecast API")

# Initialize a single ModelService instance for the API
model_service = ModelService()


@app.on_event("startup")
def startup_event():
    """
    FastAPI startup event handler.

    This function is called automatically when the FastAPI app starts.
    It loads the configuration and the trained model into the global
    `model_service` instance.

    Raises:
        Exception: If the configuration or model cannot be loaded.
    """
    try:
        model_service.load_config()
        model_service.load_model()
        print("[API] Model loaded successfully")
    except Exception as e:
        print(f"[API] Failed to load model: {e}")
        raise


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict endpoint.

    This endpoint receives historical time-series data in the request body,
    preprocesses it, extracts the required input window, and returns model
    predictions.

    Args:
        req (PredictRequest): Pydantic request model containing the `data` dictionary
                              with historical features. Example keys include "date",
                              "HUFL", "MUFL", etc.

    Returns:
        dict: A dictionary with a single key "prediction", containing a list of
              predicted values.

    Raises:
        HTTPException 500: If the model is not loaded or if prediction fails.
        HTTPException 400: If input data is invalid (e.g., too few rows).
    """
    if model_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        predictions = model_service.predict(req.data)
        return {"prediction": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API and whether the model
    is successfully loaded.

    Returns:
        dict: {
            "status": "ok",
            "model_loaded": bool - True if model is loaded, False otherwise
        }
    """
    return {"status": "ok", "model_loaded": model_service.model is not None}
