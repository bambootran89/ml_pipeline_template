import os

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from mlproject.src.models.nlinear_wrapper import FallbackNLinear
from mlproject.src.preprocess.online import online_preprocess_request

app = FastAPI(title="mlproject Forecast API")

MODEL_PATH = os.path.join("mlproject", "artifacts", "models", "model.pt")
SCALER_PATH = os.path.join("mlproject", "artifacts", "preprocessing", "scaler.pkl")

MODEL = None
SCALER = None
SCALER_COLS = None


class PredictRequest(BaseModel):
    """dict"""

    features: dict


class ModelService:
    """
    ModelService
    """

    def __init__(self):
        self.model = None


model_service = ModelService()


@app.on_event("startup")
def startup_event():
    """
    Load the scaler and PyTorch fallback model at application startup.
    """

    # Load model
    if os.path.exists(MODEL_PATH):
        input_dim = (
            model_service.scaler.shape[1] if model_service.scaler is not None else 10
        )
        model_service.model = FallbackNLinear(input_dim=input_dim, output_dim=6)
        model_service.model.load_state_dict(torch.load(MODEL_PATH))
        model_service.model.eval()


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Process input features, apply online preprocessing, and return model predictions.

    Args:
        req (PredictRequest): Request body containing a dictionary of features.

    Returns:
        dict: A dictionary with key 'prediction' containing a list of predicted values,
              or an error message if the model is not loaded.
    """
    data = req.features
    # online preprocess (fill, scale)
    processed = online_preprocess_request(
        data,
    )
    x = np.array([list(processed[c] for c in sorted(processed.keys()))], dtype=float)
    if model_service.model is None:
        return {"error": "model not loaded"}
    with torch.no_grad():
        pred = (
            model_service.model(torch.from_numpy(x.astype("float32"))).numpy().tolist()
        )
    return {"prediction": pred}
