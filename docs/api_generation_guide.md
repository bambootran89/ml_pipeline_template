# Simple API Generation Guide

Step-by-step guide to generate FastAPI and Ray Serve APIs.

---

## Prerequisites

1. You have a trained model in MLflow
2. You have a serve config file (generated from training config)

---

## Generate FastAPI

### Option 1: One Command (Generate + Run)

```bash
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/standard_train_serve.yaml
```

This will:
1. Generate `standard_train_serve_fastapi.py`
2. Start FastAPI server on `http://0.0.0.0:8000`
3. Show API docs at `http://0.0.0.0:8000/docs`

### Option 2: Generate Only

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/experiments/etth3.yaml')
gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='fastapi',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)
"
```

Output: `mlproject/serve/generated/standard_train_serve_fastapi.py`

**Then run manually:**

```bash
python mlproject/serve/generated/standard_train_serve_fastapi.py
```

---

## Generate Ray Serve

### Option 1: One Command (Generate + Run)

```bash
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml -f ray mlproject/configs/generated/standard_train_serve.yaml
```

This will:
1. Generate `standard_train_serve_ray.py`
2. Start Ray Serve with distributed replicas
3. Start API on `http://0.0.0.0:8000`
4. Ray Dashboard available at `http://localhost:8265`

### Option 2: Generate Only

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/experiments/etth3.yaml')
gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='ray',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)
"
```

Output: `mlproject/serve/generated/standard_train_serve_ray.py`

**Then run manually:**

```bash
python mlproject/serve/generated/standard_train_serve_ray.py
```

---

## Complete Example: From Training to API

### Step 1: Train Model

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_train.yaml
```

### Step 2: Generate Serve Config

```bash
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/pipelines/standard_train.yaml \
    --output-dir mlproject/configs/generated \
    --alias latest
```

This creates: `mlproject/configs/generated/standard_train_serve.yaml`

### Step 3: Generate and Run FastAPI

```bash
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/standard_train_serve.yaml
```

### Step 4: Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", ...],
      "HUFL": [5.827, 5.8, 5.969, ...],
      "MUFL": [1.599, 1.492, 1.492, ...],
      "mobility_inflow": [1.234, 1.456, 1.678, ...]
    }
  }'
```

See `docs/api_examples.md` for complete test payloads.

---

## FastAPI vs Ray Serve

| Feature | FastAPI | Ray Serve |
|---------|---------|-----------|
| **Use Case** | Simple APIs, development | Production, distributed systems |
| **Replicas** | Single process | Multiple distributed replicas |
| **Scaling** | Manual (via uvicorn workers) | Auto-scaling built-in |
| **Dashboard** | Swagger UI at `/docs` | Ray Dashboard at `:8265` |
| **Startup Time** | Fast (~2-3 seconds) | Slower (~5-10 seconds) |
| **Memory** | Lower | Higher (Ray overhead) |
| **Best For** | Development, testing, simple prod | High-traffic production |

---

## Common Issues

### Issue: serve config not found

```bash
# Generate it first
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/pipelines/standard_train.yaml \
    --output-dir mlproject/configs/generated
```

### Issue: Model not found in MLflow

Check:
1. Model was trained and logged to MLflow
2. MLflow tracking URI is correct: `echo $MLFLOW_TRACKING_URI`
3. Model has alias "latest" or "production"

```bash
# List models
mlflow models list
```

### Issue: Permission denied on serve_api.sh

```bash
chmod +x mlproject/serve_api.sh
```

---

## All Available Pipelines

You can generate APIs for any of these pipelines:

```bash
# Standard
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/standard_train_serve.yaml

# Conditional Branch
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/conditional_branch_serve.yaml

# KMeans + XGBoost
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/kmeans_then_xgboost_serve.yaml

# Parallel Ensemble
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/parallel_ensemble_serve.yaml

# Nested Sub-pipeline
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/nested_suppipeline_serve.yaml

# Dynamic Adapter
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/dynamic_adapter_train_serve.yaml
```

Replace `mlproject/serve_api.sh` with `mlproject/serve_api.sh -f ray` for Ray Serve.

---

## See Also

- [Complete API Documentation](readme_api.md)
- [API Testing Examples](api_examples.md)
- [Pipeline Generation Guide](generating_configs.md)
