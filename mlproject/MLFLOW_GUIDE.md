# MLflow Integration Guide

Hướng dẫn sử dụng MLflow để quản lý experiments, artifacts và deployment.

---

## 1. Cài đặt

```bash
pip install -r requirements.txt
```

---

## 2. Khởi động MLflow UI

```bash
# Start MLflow tracking server
mlflow ui --backend-store-uri mlruns --port 5000
```

Truy cập: http://localhost:5000

---

## 3. Training với MLflow

### 3.1. Training cơ bản

```bash
python -m mlproject.src.pipeline.run_pipeline_mlflow train \
    --config mlproject/configs/experiments/etth1.yaml
```

MLflow sẽ tự động:
- Track parameters (model type, hyperparameters)
- Log metrics (MAE, MSE, SMAPE, etc.)
- Log artifacts (model, scaler, config)
- Register model vào Model Registry

### 3.2. Xem kết quả

1. Mở MLflow UI: http://localhost:5000
2. Chọn experiment "timeseries_forecasting"
3. So sánh các runs, metrics, parameters

---

## 4. Model Registry

### 4.1. List models

```bash
bash mlproject/scripts/mlflow_commands.sh list
```

### 4.2. Promote model lên Production

```bash
# Promote version cụ thể
bash mlproject/scripts/mlflow_commands.sh promote ts_forecast_model 1

# Hoặc dùng Python
python -c "
import mlflow
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name='nlinear_forecaster',
    version=1,
    stage='Production'
)
"
```

### 4.3. Compare experiments

```bash
bash mlproject/scripts/mlflow_commands.sh compare
```

---

## 5. Evaluation từ Registry

```bash
# Evaluate latest version
python -m mlproject.src.pipeline.run_pipeline_mlflow eval \
    --config mlproject/configs/experiments/etth1.yaml \
    --model-name nlinear_forecaster \
    --model-version latest

# Evaluate version cụ thể
python -m mlproject.src.pipeline.run_pipeline_mlflow eval \
    --config mlproject/configs/experiments/etth1.yaml \
    --model-name nlinear_forecaster \
    --model-version 1
```

---

## 6. Model Serving

### 6.1. Serve với MLflow (built-in)

```bash
mlflow models serve \
    -m "models:/nlinear_forecaster/latest" \
    -p 8001 \
    --no-conda
```

Test:
```bash
curl http://localhost:8001/invocations \
    -H 'Content-Type: application/json' \
    -d @test_input.json
```

### 6.2. Serve với FastAPI (recommended)

```bash
uvicorn mlproject.serve.mlflow_api:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info

curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "data": {
            "date": ["2020-01-01 00:00:00", ...],
            "HUFL": [0.5, ...],
            "MUFL": [1.2, ...],
            "mobility_inflow": [10.0, ...]
        }
    }'
```

---

## 7. Workflow đầy đủ

### Step 1: Training

```bash
# Terminal 1: Start MLflow UI
mlflow ui --port 5000

# Terminal 2: Train models
python -m mlproject.src.pipeline.run_pipeline_mlflow train \
    --config mlproject/configs/experiments/etth1.yaml
```

### Step 2: Compare & Select

1. Mở http://localhost:5000
2. So sánh metrics của các runs
3. Chọn run tốt nhất
4. Promote lên Production

### Step 3: Evaluation

```bash
python -m mlproject.src.pipeline.run_pipeline_mlflow eval \
    --model-name nlinear_forecaster \
    --model-version latest
```

### Step 4: Deployment

```bash
# Option A: MLflow serve
mlflow models serve -m "models:/nlinear_forecaster/Production" -p 8001

# Option B: FastAPI (recommended)
uvicorn mlproject.serve.mlflow_api:app --port 8000
```

---

## 8. Advanced Usage

### 8.1. Programmatic API

```python
from mlproject.src.utils.mlflow_manager import MLflowManager
from mlproject.src.pipeline.config_loader import ConfigLoader

# Load config
cfg = ConfigLoader.load("mlproject/configs/experiments/etth1.yaml")

# Initialize manager
manager = MLflowManager(cfg)

# Start run
manager.start_run("my_experiment")

# Log params
manager.log_params({"model": "nlinear", "lr": 0.001})

# Log metrics
manager.log_metrics({"mae": 0.5, "mse": 0.3})

# Log model
manager.log_model(wrapper, input_example=x_sample)

# Register model
manager.register_model(
    model_uri=f"runs:/{manager.run_id}/model",
    model_name="my_model"
)

# End run
manager.end_run()
```

### 8.2. Load model từ Registry

```python
import mlflow

# Load latest version
model = mlflow.pyfunc.load_model("models:/nlinear_forecaster/latest")

# Load specific version
model = mlflow.pyfunc.load_model("models:/nlinear_forecaster/1")

# Load Production model
model = mlflow.pyfunc.load_model("models:/nlinear_forecaster/Production")

# Predict
predictions = model.predict(x_input)
```

### 8.3. Transition stages

```python
import mlflow

client = mlflow.MlflowClient()

# Staging -> Production
client.transition_model_version_stage(
    name="nlinear_forecaster",
    version=2,
    stage="Production",
    archive_existing_versions=True  # Archive old Production
)

# Production -> Archived
client.transition_model_version_stage(
    name="nlinear_forecaster",
    version=1,
    stage="Archived"
)
```

---

## 9. Troubleshooting

### Issue 1: Model không load được

```bash
# Check model registry
mlflow models list

# Check run artifacts
mlflow artifacts list -r runs/<run_id>
```

### Issue 2: Metrics không xuất hiện

```python
# Đảm bảo start_run được gọi
manager.start_run()
manager.log_metrics({"mae": 0.5})
manager.end_run()
```

### Issue 3: Artifact path không đúng

```python
# Sử dụng đúng artifact_path
manager.log_model(wrapper, artifact_path="model")  # NOT "models"
```

---

## 10. Best Practices

1. **Naming convention**: Dùng tên model descriptive (e.g., `nlinear_forecaster`, `tft_multivariate`)

2. **Versioning**: Luôn promote Production models, giữ history

3. **Tags**: Thêm tags cho runs để dễ filter
   ```python
   mlflow.set_tag("dataset", "ETTh1")
   mlflow.set_tag("experiment_type", "multivariate")
   ```

4. **Artifacts**: Log đầy đủ artifacts (model, scaler, config)

5. **Model Registry**: Dùng stages (Staging -> Production -> Archived)

6. **Monitoring**: Track metrics theo thời gian, setup alerts

---

## 11. References

- MLflow Docs: https://mlflow.org/docs/latest/index.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
- MLflow Tracking: https://mlflow.org/docs/latest/tracking.html
