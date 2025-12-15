from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from mlproject.serve.ray.ray_deploy import (
    ForecastAPI,
    ModelService,
    PredictRequest,
    PreprocessingService,
)

MOCK_CONFIG = {
    "mlflow": {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "test_exp",
        "registry": {"model_name": "test_model"},
    },
    "training": {"artifacts_dir": "/tmp/artifacts"},
    "experiment": {"hyperparams": {"input_chunk_length": 5}, "model": "xgboost"},
    "approaches": [{"model": "xgboost", "hyperparams": {}}],
}


@pytest.fixture
def mock_config_loader():
    """
    Fixture to mock ConfigLoader and provide a dummy configuration.
    """
    with patch("mlproject.serve.ray.ray_deploy.ConfigLoader") as mock_loader:
        cfg_mock = MagicMock()
        cfg_mock.training.artifacts_dir = "/tmp/artifacts"
        cfg_mock.experiment.hyperparams.get.return_value = 5
        cfg_mock.approaches = MOCK_CONFIG["approaches"]
        mock_loader.load.return_value = cfg_mock
        yield mock_loader


def test_model_service_init_local(mock_config_loader):
    """
    Test initializing ModelService in local fallback mode when MLflow is disabled.
    """
    with patch("mlproject.serve.ray.ray_deploy.MLflowManager") as mock_mlflow, patch(
        "mlproject.serve.ray.ray_deploy.ModelFactory"
    ) as mock_factory:
        mock_mlflow.return_value.enabled = False

        service = ModelService.func_or_class()

        mock_factory.load.assert_called_once()
        assert service.is_loaded()
        assert service.get_run_id() is None


def test_model_service_init_mlflow(mock_config_loader):
    """
    Test initializing ModelService from MLflow and retrieving the run ID.
    """
    with patch("mlproject.serve.ray.ray_deploy.MLflowManager") as mock_mlflow, patch(
        "mlproject.serve.ray.ray_deploy.load_model_from_registry_safe"
    ) as mock_load_safe:
        mock_mlflow.return_value.enabled = True

        mock_result = MagicMock()
        mock_result.model = MagicMock()
        mock_result.run_id = "run_123_abc"
        mock_load_safe.return_value = mock_result

        service = ModelService.func_or_class()

        mock_load_safe.assert_called_once()
        assert service.is_loaded()
        assert service.get_run_id() == "run_123_abc"


@pytest.mark.asyncio
async def test_preprocessing_service_lazy_init_remote(mock_config_loader):
    """
    Test PreprocessingService loads companion preprocessor from MLflow.
    """
    with patch(
        "mlproject.serve.ray.ray_deploy.MLflowManager"
    ) as mock_mlflow_manager, patch(
        "mlproject.serve.ray.ray_deploy.load_companion_preprocessor_from_model"
    ) as mock_load_companion:
        mock_mlflow_manager.return_value.enabled = True

        mock_model_handle = MagicMock()
        mock_model_handle.get_model.remote = AsyncMock(return_value="mock_model_obj")

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = pd.DataFrame({"a": [1, 2]})
        mock_load_companion.return_value = mock_preprocessor

        service = PreprocessingService.func_or_class()
        data = {"col1": [10, 20]}

        result = await service.preprocess(data, mock_model_handle)

        mock_model_handle.get_model.remote.assert_called_once()
        mock_load_companion.assert_called_once_with("mock_model_obj")
        assert isinstance(result, pd.DataFrame)


@pytest.mark.asyncio
async def test_preprocessing_service_fallback_local(mock_config_loader):
    """
    Test fallback to local TransformManager if MLflow companion preprocessor is unavailable.
    """
    with patch("mlproject.serve.ray.ray_deploy.TransformManager") as MockTM:
        mock_model_handle = MagicMock()
        mock_model_handle.get_model.remote = AsyncMock(return_value=None)

        service = PreprocessingService.func_or_class()

        instance_tm = MockTM.return_value
        instance_tm.transform.return_value = pd.DataFrame({"x": [1]})

        await service.preprocess({"x": [1]}, mock_model_handle)

        MockTM.assert_called_once()
        instance_tm.load.assert_called_once()


@pytest.mark.asyncio
async def test_forecast_api_predict_success():
    """
    Test successful end-to-end prediction workflow.
    """
    mock_pp_handle = MagicMock()
    df_result = pd.DataFrame(np.random.rand(10, 2), columns=["f1", "f2"])
    mock_pp_handle.preprocess.remote = AsyncMock(return_value=df_result)

    mock_model_handle = MagicMock()
    mock_model_handle.input_chunk_length = 5
    mock_model_handle.prepare_input.remote = AsyncMock(
        return_value=df_result.values[-5:][np.newaxis, :]
    )
    mock_model_handle.predict_prepared.remote = AsyncMock(return_value=[100.5, 101.2])

    api = ForecastAPI.func_or_class(mock_pp_handle, mock_model_handle)
    req = PredictRequest(data={"f1": [1] * 10, "f2": [2] * 10})

    response = await api.predict(req)

    assert "prediction" in response
    assert response["prediction"] == [100.5, 101.2]


@pytest.mark.asyncio
async def test_forecast_api_input_too_short():
    """
    Test HTTP 400 error when input length is shorter than input_chunk_length.
    """
    df_short = pd.DataFrame(np.random.rand(3, 2))
    mock_pp_handle = MagicMock()
    mock_pp_handle.preprocess.remote = AsyncMock(return_value=df_short)

    mock_model_handle = MagicMock()
    mock_model_handle.prepare_input.remote = AsyncMock(
        side_effect=ValueError("Need at least 5 rows")
    )

    api = ForecastAPI.func_or_class(mock_pp_handle, mock_model_handle)
    req = PredictRequest(data={"x": [1, 2, 3]})

    with pytest.raises(HTTPException) as excinfo:
        await api.predict(req)

    assert excinfo.value.status_code == 400
    assert "Need at least 5 rows" in str(excinfo.value.detail)


@pytest.mark.asyncio
async def test_forecast_api_model_failure():
    """
    Test HTTP 500 error when model.predict fails.
    """
    df = pd.DataFrame(np.zeros((10, 1)))
    mock_pp_handle = MagicMock()
    mock_pp_handle.preprocess.remote = AsyncMock(return_value=df)

    mock_model_handle = MagicMock()
    mock_model_handle.prepare_input.remote = AsyncMock(
        return_value=df.values[-5:][np.newaxis, :]
    )
    mock_model_handle.predict_prepared.remote = AsyncMock(
        side_effect=RuntimeError("GPU OOM")
    )

    api = ForecastAPI.func_or_class(mock_pp_handle, mock_model_handle)
    req = PredictRequest(data={"x": [0] * 10})

    with pytest.raises(HTTPException) as excinfo:
        await api.predict(req)

    assert excinfo.value.status_code == 500
    assert "Prediction failed" in str(excinfo.value.detail)
