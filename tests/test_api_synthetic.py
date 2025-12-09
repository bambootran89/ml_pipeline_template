"""
Test API with synthetic data - Mock model to avoid real artifacts.

Run:
    pytest mlproject/tests/test_api_synthetic.py -v -s
"""
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


def generate_synthetic_data(n_rows: int = 24) -> Dict[str, List[Any]]:
    """
    Generate synthetic data based on the pattern of real samples.

    Args:
        n_rows: Number of timesteps to generate.

    Returns:
        dict: Data dict with keys [date, HUFL, MUFL, mobility_inflow].
    """
    start_date = datetime(2020, 1, 1, 0, 0, 0)

    dates = []
    hufl_values = []
    mufl_values = []
    mobility_values = []

    for i in range(n_rows):
        dates.append((start_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"))
        hufl_values.append(round(random.uniform(-0.5, 0.5), 6))
        mufl_values.append(round(random.uniform(0.8, 1.2), 6))
        mobility_values.append(round(random.uniform(0.5, 8.0), 6))

    return {
        "date": dates,
        "HUFL": hufl_values,
        "MUFL": mufl_values,
        "mobility_inflow": mobility_values,
    }


@pytest.fixture
def mock_model_service():
    """
    Mock ModelService with fake model and config.
    This fixture is automatically applied to all tests.
    """
    from mlproject.serve.api import model_service

    # Mock config
    mock_cfg = MagicMock()
    mock_cfg.experiment.hyperparams.input_chunk_length = 24
    model_service.cfg = mock_cfg
    model_service.input_chunk_length = 24

    # Mock model wrapper
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    model_service.model = mock_model

    # Mock preprocess - return simple DataFrame
    def mock_preprocess(data_dict):
        df = pd.DataFrame(data_dict)
        if "date" in df.columns:
            df = df.set_index("date")
        return pd.DataFrame(
            np.random.randn(len(df), 3), columns=["feature1", "feature2", "feature3"]
        )

    model_service.preprocess = mock_preprocess

    # Mock prepare_input_window
    def mock_prepare_window(df):
        n_features = df.shape[1]
        return np.random.randn(1, 24, n_features)

    model_service.prepare_input_window = mock_prepare_window

    yield model_service


@pytest.fixture
def client(mock_model_service):
    """FastAPI test client with mocked model service."""
    from mlproject.serve.api import app

    return TestClient(app)


@pytest.fixture
def synthetic_24_rows():
    """Generate 24 rows of synthetic data."""
    return {"data": generate_synthetic_data(n_rows=24)}


@pytest.fixture
def synthetic_48_rows():
    """Generate 48 rows of synthetic data."""
    return {"data": generate_synthetic_data(n_rows=48)}


@pytest.fixture
def synthetic_10_rows():
    """Generate 10 rows of synthetic data."""
    return {"data": generate_synthetic_data(n_rows=10)}


class TestSyntheticData:
    """Tests using synthetic data."""

    def test_predict_exact_24_rows(self, client, synthetic_24_rows):
        """Test with exactly 24 rows."""
        response = client.post("/predict", json=synthetic_24_rows)

        print(f"\nStatus: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert isinstance(data["prediction"], list)
        assert len(data["prediction"]) == 6  # Mocked output

        predictions = data["prediction"]
        print(f"Predictions: {predictions}")

    def test_predict_more_than_24_rows(self, client, synthetic_48_rows):
        """Test with 48 rows (API uses last 24)."""
        response = client.post("/predict", json=synthetic_48_rows)

        print(f"\nStatus: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        predictions = data["prediction"]
        print(f"Predictions: {predictions}")

    def test_predict_less_than_24_rows(
        self, client, synthetic_10_rows, mock_model_service
    ):
        """Test with 10 rows (insufficient)."""

        # Override mock to raise error
        def mock_prepare_error(df):
            raise ValueError("Input data has 10 rows, need at least 24")

        mock_model_service.prepare_input_window = mock_prepare_error

        response = client.post("/predict", json=synthetic_10_rows)

        print(f"\nStatus: {response.status_code}")
        print(f"Response: {response.json()}")

        assert response.status_code == 400
        assert "need at least" in response.json()["detail"].lower()

    def test_multiple_predictions_consistency(self, client, synthetic_24_rows):
        """Test consistency: calling twice with same data."""
        response1 = client.post("/predict", json=synthetic_24_rows)
        response2 = client.post("/predict", json=synthetic_24_rows)

        assert response1.status_code == 200
        assert response2.status_code == 200

        pred1 = response1.json()["prediction"]
        pred2 = response2.json()["prediction"]

        # With deterministic mock, predictions should be identical
        assert pred1 == pred2
        print(f"\nConsistent predictions: {pred1}")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_zeros(self, client):
        """Test with all features = 0."""
        zero_data = {
            "data": {
                "date": [f"2020-01-01 {i:02d}:00:00" for i in range(24)],
                "HUFL": [0.0] * 24,
                "MUFL": [0.0] * 24,
                "mobility_inflow": [0.0] * 24,
            }
        }

        response = client.post("/predict", json=zero_data)

        assert response.status_code == 200
        predictions = response.json()["prediction"]
        print(f"\nZero input predictions: {predictions}")

    def test_negative_values(self, client):
        """Test with negative values."""
        negative_data = {
            "data": {
                "date": [f"2020-01-01 {i:02d}:00:00" for i in range(24)],
                "HUFL": [-1.0] * 24,
                "MUFL": [-1.0] * 24,
                "mobility_inflow": [-1.0] * 24,
            }
        }

        response = client.post("/predict", json=negative_data)

        assert response.status_code == 200
        predictions = response.json()["prediction"]
        print(f"\nNegative input predictions: {predictions}")


class TestRandomSamples:
    """Tests with multiple random samples."""

    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_different_random_seeds(self, client, seed):
        """Test with different random seeds."""
        random.seed(seed)
        data = {"data": generate_synthetic_data(n_rows=24)}

        response = client.post("/predict", json=data)

        assert response.status_code == 200
        predictions = response.json()["prediction"]
        print(f"\nSeed {seed} predictions: {predictions}")

    def test_batch_predictions(self, client):
        """Test consecutive multiple calls."""
        results = []

        for i in range(5):
            random.seed(i)
            data = {"data": generate_synthetic_data(n_rows=24)}
            response = client.post("/predict", json=data)

            assert response.status_code == 200
            predictions = response.json()["prediction"]
            results.append(predictions)

        print(f"\nBatch predictions (5 calls):")
        for i, pred in enumerate(results):
            print(f"  Call {i+1}: {pred}")


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        print(f"\nHealth: {data}")


# Manual test script (requires API running)
if __name__ == "__main__":
    import json

    import requests

    print("=" * 60)
    print("MANUAL API TEST (requires running API)")
    print("=" * 60)

    # Test 1: Health
    print("\n[1] Health Check:")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure API is running: uvicorn mlproject.serve.api:app --reload")
        exit(1)

    # Test 2: Prediction
    print("\n[2] Prediction with 24 rows:")
    data = {"data": generate_synthetic_data(n_rows=24)}

    try:
        response = requests.post("http://localhost:8000/predict", json=data, timeout=10)
        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"ERROR: {e}")
