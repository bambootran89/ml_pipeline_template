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
