# Serve API Generation and Running

Complete guide for generating and running serve APIs from configurations.

## Quick Start

Run API server from serve config in one command:

```bash
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

API will start at `http://localhost:8000`

---

## Prerequisites

1. Serve config file exists (example: `standard_train_serve.yaml`)
2. Script is executable: `chmod +x serve_api.sh`

## Generate Serve Config (if needed)

```bash
python -m mlproject.src.pipeline.dag_run generate \
    mlproject/configs/pipelines/standard_train.yaml \
    --config-type serve \
    --output-dir mlproject/configs/generated
```

## Command Options

### Basic usage

```bash
# FastAPI (default)
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
./serve_api.sh -f ray mlproject/configs/generated/standard_train_serve.yaml

# Custom port
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml

# All options
./serve_api.sh -f ray -p 9000 -h 127.0.0.1 mlproject/configs/generated/standard_train_serve.yaml
```

### Python method

```bash
python serve_api.py --serve-config mlproject/configs/generated/standard_train_serve.yaml
```

---

## Testing API

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": [1,2,3], "feature2": [4,5,6]}}'
```

### Swagger UI (FastAPI only)

Open browser: `http://localhost:8000/docs`

---

## Examples

### Example 1: Standard Pipeline

```bash
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

### Example 2: Conditional Branch (Multi-model)

```bash
./serve_api.sh mlproject/configs/generated/conditional_branch_serve.yaml
```

### Example 3: Ray Serve on Port 9000

```bash
./serve_api.sh -f ray -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

---

## How it Works

When you run `./serve_api.sh`, the script will:

1. Auto-generate FastAPI or Ray Serve code from serve.yaml
2. Configure host and port
3. Run API server

---

## Troubleshooting

### Port already in use

```bash
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Module not found

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

### Script not executable

```bash
# Make executable
chmod +x serve_api.sh

# Or use Python directly
python serve_api.py --serve-config mlproject/configs/generated/standard_train_serve.yaml
```

---

## See Also

- `README_API.md` - Detailed documentation
- `examples/generate_serve_apis.py` - Example scripts
