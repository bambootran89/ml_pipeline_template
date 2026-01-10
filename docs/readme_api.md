# API Generation and Running Guide

Complete guide for generating Python API code from serve configurations and running them.
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
    experiment_config_path="mlproject/configs/pipelines/standard_train.yaml"
)
print(f"FastAPI generated: {fastapi_path}")

# Generate Ray Serve
ray_path = generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="ray",
    experiment_config_path="mlproject/configs/pipelines/standard_train.yaml"
)
print(f"Ray Serve generated: {ray_path}")
```

### Method 2: Using Python Script

```bash
# Generate all APIs for a pipeline
python examples/generate_serve_apis.py
```

### Method 3: Command Line (Generate Only)

```bash
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator

gen = ConfigGenerator('mlproject/configs/pipelines/standard_train.yaml')

# FastAPI
gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='fastapi',
    experiment_config_path='mlproject/configs/pipelines/standard_train.yaml'
)

# Ray Serve
gen.generate_api(
    serve_config_path='mlproject/configs/generated/standard_train_serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='ray',
    experiment_config_path='mlproject/configs/pipelines/standard_train.yaml'
)
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
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
./serve_api.sh -f ray mlproject/configs/generated/standard_train_serve.yaml

# Custom port
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml
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

### Server Output

When server starts, you will see:

```
============================================================
Auto-Generate & Run FASTAPI API
============================================================
Serve config: mlproject/configs/generated/standard_train_serve.yaml
Train config: mlproject/configs/pipelines/standard_train.yaml
Framework: fastapi
Address: 0.0.0.0:8000
============================================================

[1/2] Generating API code...
[ApiGenerator] Generated fastapi API: mlproject/serve/generated/standard_train_serve_fastapi.py
Generated: mlproject/serve/generated/standard_train_serve_fastapi.py

[2/3] Configuring server settings...
Configured: 0.0.0.0:8000

[3/3] Starting FASTAPI server...

============================================================
API starting at: http://0.0.0.0:8000
API docs: http://0.0.0.0:8000/docs
Health check: http://0.0.0.0:8000/health
============================================================

Press Ctrl+C to stop the server

------------------------------------------------------------
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 3. Test APIs

**For complete, realistic testing examples, see:** `API_EXAMPLES.md`

The examples below use ETTh1 dataset structure from experiment configs.

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

**Note:** For ETTh1 config, input requires 24 timesteps (input_chunk_length). See `API_EXAMPLES.md` for complete examples.

#### Quick Test (simplified)

```bash
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

**Expected Response (12 predictions = 2 targets × 6 timesteps):**

```json
{
  "predictions": [
    5.628, 5.701, 5.823, 5.945, 6.078, 6.201,
    2.234, 2.267, 2.301, 2.334, 2.367, 2.401
  ]
}
```

### Using Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction with realistic data
# See API_EXAMPLES.md for complete examples or use helper:
from examples.generate_test_data import generate_test_data

test_data = generate_test_data(num_timesteps=24)
payload = {"data": test_data}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)
print(response.json())
```

### Interactive API Documentation (FastAPI only)

FastAPI auto-generates interactive Swagger UI documentation.

Open in browser: `http://localhost:8000/docs`

Features:
- View all endpoints
- Test requests interactively
- See request/response schemas
- Download OpenAPI spec

---

## 4. Examples

### Example 1: Standard Single-Model Pipeline

**Step 1: Generate serve config (if not exists)**

```bash
python -m mlproject.src.pipeline.dag_run generate \
    mlproject/configs/pipelines/standard_train.yaml \
    --config-type serve \
    --output-dir mlproject/configs/generated
```

**Step 2: Generate and run API**

```bash
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

**Step 3: Test**

See `API_EXAMPLES.md` for complete testing examples with realistic data.

```bash
# Health check
curl http://localhost:8000/health

# Prediction (see API_EXAMPLES.md for full payload)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### Example 2: Conditional Branch (Multi-Model)

**Step 1: Generate and run**

```bash
./serve_api.sh mlproject/configs/generated/conditional_branch_serve.yaml
```

**Step 2: Test**

```bash
curl http://localhost:8000/health
# See API_EXAMPLES.md for realistic test payloads
```

### Example 3: Ray Serve (Distributed)

**Step 1: Generate and run with Ray**

```bash
./serve_api.sh -f ray mlproject/configs/generated/standard_train_serve.yaml
```

**Step 2: Access Ray Dashboard**

Open browser: `http://localhost:8265`

**Step 3: Test API**

```bash
curl http://localhost:8000/health
```

### Example 4: Custom Port

```bash
# Run on port 9000
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml

# Test with custom port
curl http://localhost:9000/health
```

### Example 5: Generate Only (No Run)

```python
from mlproject.src.utils.generator.config_generator import ConfigGenerator

generator = ConfigGenerator("mlproject/configs/pipelines/standard_train.yaml")

# Generate FastAPI file
fastapi_path = generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="fastapi",
    experiment_config_path="mlproject/configs/pipelines/standard_train.yaml"
)

print(f"Generated: {fastapi_path}")
# File is created but not running yet
# You can edit it before running
```

### Example 6: Load Testing

See `API_EXAMPLES.md` Example 4 for complete load testing guide with realistic data.

```bash
# Install apache bench
sudo apt-get install apache2-utils

# Create test payload (see API_EXAMPLES.md for full payload)
# request.json should contain 24 timesteps of ETTh1 data

# Run 1000 requests with 10 concurrent connections
ab -n 1000 -c 10 -T 'application/json' \
   -p request.json \
   http://localhost:8000/predict
```

---

## Troubleshooting

### Port Already in Use

```bash
# Use different port
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Module Not Found

```bash
# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Then run
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

### Model Not Loading

Check:
1. MLflow tracking URI: `echo $MLFLOW_TRACKING_URI`
2. Model exists: `mlflow models list`
3. Alias is correct (default: "production")

```bash
# Set MLflow URI if needed
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Import Errors

```bash
# Install dependencies
pip install fastapi uvicorn ray[serve] pandas numpy
```

---

## Command Reference

### Generate API Code

```bash
# FastAPI
python -c "from mlproject.src.utils.generator.config_generator import ConfigGenerator; \
           gen = ConfigGenerator('train.yaml'); \
           gen.generate_api(serve_config_path='serve.yaml', \
                           output_dir='output/', \
                           framework='fastapi', \
                           experiment_config_path='train.yaml')"

# Ray Serve
python -c "from mlproject.src.utils.generator.config_generator import ConfigGenerator; \
           gen = ConfigGenerator('train.yaml'); \
           gen.generate_api(serve_config_path='serve.yaml', \
                           output_dir='output/', \
                           framework='ray', \
                           experiment_config_path='train.yaml')"
```

### Run API

```bash
# Quick run (auto-generate + run)
./serve_api.sh <serve_config.yaml>

# With options
./serve_api.sh -f <fastapi|ray> -p <port> -h <host> <serve_config.yaml>

# Python method
python serve_api.py --serve-config <serve_config.yaml> \
                    --framework <fastapi|ray> \
                    --port <port>

# Run pre-generated file
python mlproject/serve/generated/<pipeline>_fastapi.py
python mlproject/serve/generated/<pipeline>_ray.py
```

### Test API

```bash
# Health
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {...}}'

# Docs (FastAPI only)
open http://localhost:8000/docs
```

---

## See Also

- `QUICK_START.md` - Quick reference guide
- `examples/generate_serve_apis.py` - Example generation script
- `mlproject/serve/generated/README.md` - Generated files documentation
