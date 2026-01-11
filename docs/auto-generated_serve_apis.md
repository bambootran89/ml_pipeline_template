# Auto-Generated Serve APIs

This directory contains auto-generated API code for serving ML models.

## Overview

APIs are automatically generated from serve pipeline configurations using `ConfigGenerator.generate_api()`. Two frameworks are supported:

- **FastAPI**: Lightweight, synchronous REST API
- **Ray Serve**: Distributed, scalable microservices architecture

## Generated Files

### FastAPI Examples
- `standard_train_serve_fastapi.py` - Simple single-model API
- `conditional_branch_serve_fastapi.py` - Multi-model conditional API

### Ray Serve Examples
- `standard_train_serve_ray.py` - Distributed single-model API
- `conditional_branch_serve_ray.py` - Distributed multi-model API

## Usage

### Generating APIs

```python
from mlproject.src.utils.generator.config_generator import ConfigGenerator

# Initialize generator
generator = ConfigGenerator("mlproject/configs/pipelines/standard_train.yaml")

# Generate FastAPI
generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="fastapi",
    experiment_config_path="mlproject/configs/pipelines/standard_train.yaml"
)

# Generate Ray Serve
generator.generate_api(
    serve_config_path="mlproject/configs/generated/standard_train_serve.yaml",
    output_dir="mlproject/serve/generated",
    framework="ray",
    experiment_config_path="mlproject/configs/pipelines/standard_train.yaml"
)
```

### Running Generated APIs

#### FastAPI

```bash
# Run directly
python mlproject/serve/generated/standard_train_serve_fastapi.py

# Or with uvicorn
uvicorn mlproject.serve.generated.standard_train_serve_fastapi:app --reload
```

Access at: http://localhost:8000

API endpoints:
- `GET /health` - Health check
- `POST /predict` - Make predictions

#### Ray Serve

```bash
# Run the generated script
python mlproject/serve/generated/standard_train_serve_ray.py
```

Access at: http://localhost:8000

Ray Dashboard: http://localhost:8265

## API Structure

### FastAPI Structure

```python
# Single service class
class ServeService:
    def __init__(self, config_path: str)
    def preprocess(self, data: pd.DataFrame)
    def predict(self, features: pd.DataFrame, model_key: str)

# Endpoints
@app.get("/health")
@app.post("/predict")
```

### Ray Serve Structure

```python
# Microservices
@serve.deployment
class ModelService:
    async def predict(...)

@serve.deployment
class PreprocessService:
    async def preprocess(...)

@serve.deployment
@serve.ingress(app)
class ServeAPI:
    @app.post("/predict")
    async def predict(...)
```

## Customization

Generated code is meant to be a starting point. You can:

1. **Modify request/response schemas** - Add custom fields
2. **Add authentication** - Integrate with your auth system
3. **Add logging** - Integrate with your logging infrastructure
4. **Tune performance** - Adjust Ray Serve replicas, resources
5. **Add endpoints** - Add custom business logic endpoints

## Configuration

### Models

Models are loaded from MLflow based on the serve config's `load_map`:

```yaml
- id: init_artifacts
  type: mlflow_loader
  load_map:
    - step_id: train_model
      context_key: fitted_train_model
```

This generates:

```python
self.models["fitted_train_model"] = mlflow_manager.load_component(
    name=f"{experiment_name}_train_model",
    alias="production"
)
```

### Preprocessors

If a preprocessor step exists in the serve config:

```yaml
- id: preprocess
  type: preprocessor
  instance_key: fitted_preprocess
```

It's automatically loaded and used before inference.

## Multi-Model Support

For conditional branches with multiple models:

```yaml
- id: model_selection
  type: branch
  if_true:
    id: train_tft_inference
    type: inference
    wiring:
      inputs:
        model: fitted_train_tft
  if_false:
    id: train_xgb_inference
    type: inference
    wiring:
      inputs:
        model: fitted_train_xgb
```

All models are loaded:

```python
self.models["fitted_train_tft"] = ...
self.models["fitted_train_xgb"] = ...
```

## Regeneration

To regenerate APIs after config changes:

```bash
# Re-run the generation command
python -c "
from mlproject.src.utils.generator.config_generator import ConfigGenerator
gen = ConfigGenerator('train_config.yaml')
gen.generate_api(
    serve_config_path='serve.yaml',
    output_dir='mlproject/serve/generated',
    framework='fastapi'
)
"
```

## Best Practices

1. **Version control generated code** - Track changes in git
2. **Review before deploying** - Check generated code makes sense
3. **Test locally first** - Verify health and predict endpoints
4. **Monitor in production** - Add metrics and alerts
5. **Keep configs in sync** - Regenerate when serve config changes

## Troubleshooting

### Model not loading

- Check MLflow tracking URI is set
- Verify model alias exists ("production")
- Check experiment name matches config

### Import errors

- Ensure all dependencies installed: `pip install fastapi uvicorn ray[serve]`
- Check Python path includes mlproject

### Port already in use

- FastAPI: Change port in `uvicorn.run(..., port=8000)`
- Ray: Ray Serve uses port 8000 by default, dashboard uses 8265

## Next Steps

- Add batch prediction endpoints
- Integrate with Feast for online features
- Add request validation and rate limiting
- Deploy to Kubernetes with proper scaling
- Add A/B testing support for multiple models
