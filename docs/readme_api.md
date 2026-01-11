# API Generation Guide

This guide explains how to generate and test FastAPI/Ray Serve APIs from ML pipeline configurations.

## Overview

The API generator supports two data types:
- **Tabular data**: Batch prediction for multiple rows
- **Timeseries data**: Multi-step prediction for forecasting

## Quick Start

### 1. Generate APIs

```bash
# Generate APIs for all pipelines (timeseries + tabular)
python examples/generate_serve_apis.py
```

Generated files will be in `mlproject/serve/generated/`:
- `*_fastapi.py` - FastAPI standalone server
- `*_ray.py` - Ray Serve distributed deployment

### 2. Run the API

**FastAPI (Timeseries):**
```bash
python mlproject/serve/generated/standard_train_serve_fastapi.py
```

**FastAPI (Tabular):**
```bash
python mlproject/serve/generated/tabular_serve_fastapi.py
```

**Ray Serve:**
```bash
python mlproject/serve/generated/standard_train_serve_ray.py
```

## API Endpoints

### Common Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Standard prediction |

### Tabular-specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/batch` | POST | Batch prediction with optional probabilities |

### Timeseries-specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/multistep` | POST | Multi-step prediction with configurable horizon |

## Testing with curl

### Health Check

```bash
curl http://localhost:8000/health
```

### Tabular Predictions

**Single prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": [3],
      "Age": [25.0],
      "SibSp": [0],
      "Parch": [0],
      "Fare": [7.25],
      "Sex": [1],
      "Embarked": [2]
    }
  }'
```

**Batch prediction (10 samples):**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
      "Age": [22.0, 38.0, 26.0, 35.0, 28.0, 45.0, 19.0, 55.0, 32.0, 41.0],
      "SibSp": [1, 1, 0, 1, 0, 0, 3, 0, 1, 0],
      "Parch": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.05, 21.07, 30.5, 15.55, 26.55],
      "Sex": [1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
      "Embarked": [2, 0, 2, 2, 2, 1, 2, 2, 2, 2]
    },
    "return_probabilities": true
  }'
```

### Timeseries Predictions

**Important:** All arrays must have the same length (24 timesteps for input_chunk_length=24).

**Standard prediction (output_chunk_length steps):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    }
  }'
```

**Multi-step prediction (18 steps = 3 blocks of 6):**
```bash
curl -X POST http://localhost:8000/predict/multistep \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    },
    "steps_ahead": 18
  }'
```

**Using Python to generate test data (recommended):**
```python
# This generates properly formatted test data
python -c "
from examples.generate_test_data import generate_timeseries_data
import json
data = generate_timeseries_data(num_timesteps=24)
print(json.dumps({'data': data}))
"
```

## Testing with Python

### Generate Test Data

```python
from examples.generate_test_data import (
    generate_tabular_data,
    generate_timeseries_data,
    generate_long_timeseries_data,
)

# Tabular data (10 samples)
tabular_data = generate_tabular_data(num_samples=10)

# Timeseries data (24 timesteps)
ts_data = generate_timeseries_data(num_timesteps=24)

# Long timeseries for multi-step prediction
long_ts = generate_long_timeseries_data(
    input_chunk_length=24,
    output_chunk_length=6,
    num_prediction_blocks=3,
)
```

### Tabular Batch Prediction

```python
import requests
from examples.generate_test_data import generate_tabular_data

# Generate test data
data = generate_tabular_data(num_samples=50)

# Batch prediction with probabilities
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "data": data,
        "return_probabilities": True
    }
)

result = response.json()
print(f"Predictions: {result['predictions']}")
print(f"Metadata: {result['metadata']}")
# Output:
# Predictions: {'inference_predictions': [0, 1, 1, ...]}
# Metadata: {'n_samples': 50, 'model_type': 'tabular'}
```

### Timeseries Multi-step Prediction

```python
import requests
from examples.generate_test_data import generate_timeseries_data

# Generate input data (at least input_chunk_length timesteps)
data = generate_timeseries_data(num_timesteps=24)

# Multi-step prediction (18 steps = 3 prediction blocks)
response = requests.post(
    "http://localhost:8000/predict/multistep",
    json={
        "data": data,
        "steps_ahead": 18  # 3 blocks * 6 output_chunk_length
    }
)

result = response.json()
print(f"Predictions: {len(result['predictions']['inference_predictions'])} steps")
print(f"Metadata: {result['metadata']}")
# Output:
# Predictions: 18 steps
# Metadata: {'steps_ahead': 18, 'output_chunk_length': 6, 'n_blocks': 3, 'model_type': 'timeseries'}
```

## Response Format

### Success Response

```json
{
  "predictions": {
    "inference_predictions": [0.5, 0.6, 0.7, ...]
  },
  "metadata": {
    "n_samples": 10,
    "model_type": "tabular"
  }
}
```

### Tabular with Probabilities

```json
{
  "predictions": {
    "inference_predictions": [0, 1, 1, 0, ...],
    "inference_predictions_probabilities": [[0.8, 0.2], [0.3, 0.7], ...]
  },
  "metadata": {
    "n_samples": 10,
    "model_type": "tabular"
  }
}
```

### Timeseries Multi-step

```json
{
  "predictions": {
    "inference_predictions": [5.1, 5.2, 5.3, ..., 6.0]
  },
  "metadata": {
    "steps_ahead": 18,
    "output_chunk_length": 6,
    "n_blocks": 3,
    "model_type": "timeseries"
  }
}
```

## Configuration

The API generator extracts configuration from the experiment YAML:

```yaml
data:
  type: tabular  # or "timeseries"
  features: ["Pclass", "Age", "SibSp", ...]

experiment:
  hyperparams:
    input_chunk_length: 24   # For timeseries
    output_chunk_length: 6   # For timeseries
```

## Error Handling

### Invalid Input Length (Timeseries)

```json
{
  "detail": "Input must have at least 24 timesteps (got 10)"
}
```

### Model Not Found

```json
{
  "detail": "Model not loaded"
}
```

## File Structure

```
mlproject/
├── serve/
│   └── generated/
│       ├── standard_train_serve_fastapi.py  # Timeseries FastAPI
│       ├── standard_train_serve_ray.py      # Timeseries Ray Serve
│       ├── tabular_serve_fastapi.py         # Tabular FastAPI
│       └── tabular_serve_ray.py             # Tabular Ray Serve
├── configs/
│   ├── experiments/
│   │   ├── tabular.yaml                     # Tabular experiment config
│   │   └── etth1.yaml                       # Timeseries experiment config
│   └── generated/
│       ├── tabular_train.yaml               # Merged tabular config
│       └── tabular_serve.yaml               # Generated serve config
└── examples/
    ├── generate_serve_apis.py               # API generation script
    └── generate_test_data.py                # Test data generation
```

## Customization

### Adding New Model Types

Edit `mlproject/src/utils/generator/api_generator_mixin.py`:

```python
def _infer_model_type(self, model_key: str) -> str:
    ml_patterns = ["xgboost", "catboost", "your_new_model", ...]
    dl_patterns = ["tft", "nlinear", "your_dl_model", ...]
    # ...
```

### Custom Features

The generator reads features from the config:

```yaml
data:
  features: ["custom_feature_1", "custom_feature_2", ...]
```

# API Generation and Running Guide

Complete guide for generating Python API code from serve configurations and running them.

**New to API generation?** See [Simple API Generation Guide](api_generation_guide.md) for a quick start with FastAPI and Ray Serve.

---

## Quick Start

### Step 1: Generate Serve Config (if not exists)

```bash
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/pipelines/standard_train.yaml \
    --output-dir mlproject/configs/generated \
    --alias latest
```

This generates:
- `standard_train_eval.yaml`
- `standard_train_serve.yaml` (we need this)
- `standard_train_tune.yaml`

### Step 2: Generate and Run FastAPI

```bash
# Method 1: Auto-generate and run in one command (EASIEST)
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/standard_train_serve.yaml

# Method 2: Generate only, then run manually
python -m mlproject.serve.run_generated_api \
    --experiment-config mlproject/configs/experiments/etth3.yaml \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --framework fastapi \
    --port 8000
```

### Step 3: Generate and Run Ray Serve

```bash
# Method 1: Auto-generate and run in one command (EASIEST)
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml -f ray mlproject/configs/generated/standard_train_serve.yaml

# Method 2: Generate only, then run manually
python -m mlproject.serve.run_generated_api \
    --experiment-config mlproject/configs/experiments/etth3.yaml \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --framework ray \
    --port 8000
```

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Prediction (see docs/api_examples.md for complete examples)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

---

## 1. Generate Python API Files

### Method 1: Using ConfigGenerator (Python)

```python
from mlproject.src.utils.generator.config_generator import ConfigGenerator

# Initialize generator with training config
generator = ConfigGenerator("mlproject/configs/pipelines/standard_train.yaml")

# Generate FastAPI
fastapi_path = generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="fastapi",
    experiment_config_path="mlproject/configs/experiments/etth3.yaml"
)
print(f"FastAPI generated: {fastapi_path}")

# Generate Ray Serve
ray_path = generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="ray",
    experiment_config_path="mlproject/configs/experiments/etth3.yaml"
)
print(f"Ray Serve generated: {ray_path}")
```

### Method 2: Using Python Script

```bash
# Generate all APIs for a pipeline
python examples/generate_serve_apis.py
```

### Method 3: Command Line (Generate Only)

#### Generate FastAPI Only

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/experiments/etth3.yaml')

# Generate FastAPI file
api_path = gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='fastapi',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)
print(f'Generated FastAPI: {api_path}')
"
```

Output: `mlproject/serve/generated/standard_train_serve_fastapi.py`

#### Generate Ray Serve Only

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/experiments/etth3.yaml')

# Generate Ray Serve file
api_path = gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='ray',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)
print(f'Generated Ray Serve: {api_path}')
"
```

Output: `mlproject/serve/generated/standard_train_serve_ray.py`

#### Generate Both FastAPI and Ray Serve

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/experiments/etth3.yaml')

# Generate both
fastapi = gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='fastapi',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)

ray = gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='ray',
    experiment_config_path='mlproject/configs/experiments/etth3.yaml'
)

print(f'Generated:')
print(f'  FastAPI: {fastapi}')
print(f'  Ray Serve: {ray}')
"
```

### Generated Files Location

All generated Python files are saved to: `mlproject/serve/generated/`

Example output:
```
mlproject/serve/generated/
├── standard_train_serve_fastapi.py
├── standard_train_serve_ray.py
├── conditional_branch_serve_fastapi.py
├── conditional_branch_serve_ray.py
└── ...
```

---

## 2. Run APIs

### Method 1: Auto-generate and Run (Recommended)

```bash
# FastAPI (auto-generates code and runs immediately)
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml -f ray mlproject/configs/generated/standard_train_serve.yaml

# Custom port
mlproject/serve_api.sh -e mlproject/configs/experiments/etth3.yaml -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Method 2: Run Pre-generated Files

#### FastAPI

```bash
# Direct execution
python mlproject/serve/generated/standard_train_serve_fastapi.py

# Using uvicorn (recommended for production)
uvicorn mlproject.serve.generated.standard_train_serve_fastapi:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
```

#### Ray Serve

```bash
# Direct execution
python mlproject/serve/generated/standard_train_serve_ray.py
```

### Method 3: Python Module

```bash
python -m mlproject.serve.run_generated_api \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --framework fastapi \
    --port 8000
```

---

## 3. Test APIs

**For complete, realistic testing examples, see:** `API_EXAMPLES.md`

The examples below use ETTh3 dataset structure from experiment configs.

### Health Check

#### Using curl

```bash
curl http://localhost:8000/health
```

#### Expected Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction

**Note:** For ETTh3 config, input requires 24 timesteps (input_chunk_length). See `API_EXAMPLES.md` for complete examples.

#### Quick Test (simplified)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "HUFL": [-0.15, 0.08, 0.01, -0.01, 0.21, -0.15, 0.12, 0.05, -0.08, 0.18, -0.12, 0.22, 0.03, -0.18, 0.15, -0.05, 0.28, -0.22, 0.08, -0.15, 0.32, -0.28, 0.12, -0.18],
      "MUFL": [1.14, 1.06, 0.93, 1.11, 0.96, 1.05, 0.98, 1.12, 0.95, 1.08, 0.92, 1.15, 0.88, 1.22, 0.85, 1.18, 0.82, 1.25, 0.78, 1.32, 0.75, 1.38, 0.72, 1.42],
      "mobility_inflow": [1.24, 4.42, 7.28, 1.03, 0.73, 2.5, 3.2, 4.1, 1.8, 5.3, 2.1, 6.4, 1.5, 7.8, 3.6, 4.9, 2.7, 8.2, 1.9, 5.5, 3.8, 6.7, 2.3, 4.2]
    }
  }'

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

list processes and kill

lsof -nP -iTCP:8000 -sTCP:LISTEN
