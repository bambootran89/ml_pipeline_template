"""
FastAPI serving model từ MLflow Model Registry.
"""
from fastapi import FastAPI, HTTPException

from mlproject.serve.mlflow_service import MLflowModelService
from mlproject.serve.schemas import PredictRequest

app = FastAPI(title="MLflow Model Serving API")

# Initialize model service
model_service = MLflowModelService(
    model_name="ts_forecast_model",
    model_version="latest",  # Có thể thay bằng version cụ thể
)


@app.on_event("startup")
def startup_event():
    """
    Load model từ MLflow Registry khi API khởi động.
    """
    try:
        model_service.load_model()
        print("[API] Model loaded from MLflow Registry")
    except Exception as e:
        print(f"[API] Failed to load model: {e}")
        raise


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Prediction endpoint.

    Args:
        req: Request chứa input data

    Returns:
        dict: {"prediction": [values...]}
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

    Returns:
        dict: Status và model info
    """
    return {
        "status": "ok",
        "model_loaded": model_service.model is not None,
        "model_name": model_service.model_name,
        "model_version": model_service.model_version,
    }


@app.get("/model-info")
def model_info():
    """
    Get thông tin về model hiện tại.

    Returns:
        dict: Model metadata
    """
    if model_service.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_name": model_service.model_name,
        "model_version": model_service.model_version,
        "input_chunk_length": model_service.input_chunk_length,
    }
