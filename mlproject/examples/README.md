# ConfigGenerator Examples

This directory contains examples demonstrating how to use the `ConfigGenerator` utility to automatically generate evaluation and serving pipeline configurations.

## Overview

The `ConfigGenerator` automates the transformation of training pipeline configs into eval/serve configs by:

- Loading artifacts from MLflow registry
- Transforming preprocessing steps to load mode
- Generating evaluator/inference steps
- Setting up correct dependencies and wiring

## Examples

### 1. Nested Sub-Pipeline Support

**File:** `generate_configs_for_nested_pipeline.py`

Demonstrates config generation for pipelines with nested sub-pipelines:

```bash
python mlproject/examples/generate_configs_for_nested_pipeline.py
```

**Supported Pipeline Types:**
- ✓ Standard flat pipelines (`standard_train.yaml`)
- ✓ Nested sub-pipelines (`nested_suppipeline.yaml`)
- ✓ Two-stage pipelines (`kmeans_then_xgboost.yaml`)
- ✓ Parallel ensemble pipelines (`parallel_ensemble.yaml`)

**Key Features:**
- Recursively extracts model producers and preprocessors from nested structures
- Preserves sub-pipeline architecture (no flattening)
- Transforms internal steps for eval/serve modes
- Generates correct MLflow loader configurations
- Handles complex wiring and dependencies

## Usage Pattern

```python
from mlproject.src.utils.config_generator import ConfigGenerator

# Initialize with training config
generator = ConfigGenerator("path/to/train_config.yaml")

# Generate eval config
generator.generate_eval_config(
    alias="latest",
    output_path="path/to/eval_config.yaml"
)

# Generate serve config
generator.generate_serve_config(
    alias="latest",
    output_path="path/to/serve_config.yaml"
)

# Or generate both at once
results = generator.generate_all(
    output_dir="path/to/output",
    alias="production"
)
```

## Generated Config Structure

### Eval Config

```yaml
pipeline:
  steps:
    - data_loader          # Load test data
    - mlflow_loader        # Restore artifacts (models + preprocessors)
    - sub_pipeline         # Transformed for eval mode (if exists)
    - evaluator(s)         # One per model producer
    - profiling            # Output statistics
    - logger               # Log results
```

### Serve Config

```yaml
pipeline:
  steps:
    - mlflow_loader        # Restore artifacts
    - sub_pipeline         # Transformed for serve mode (if exists)
    - inference step(s)    # Generate predictions
    - profiling            # Output statistics
```

## Advanced Features

### Nested Sub-Pipeline Transformation

When a training pipeline contains sub-pipelines:

**Training Mode:**
```yaml
- id: feature_pipeline
  type: sub_pipeline
  pipeline:
    steps:
      - id: normalize
        type: preprocessor
        is_train: true        # Fit on training data
        log_artifact: true
      - id: cluster
        type: clustering
        log_artifact: true
        hyperparams:
          n_clusters: 4
```

**Eval Mode (Auto-generated):**
```yaml
- id: feature_pipeline
  type: sub_pipeline
  pipeline:
    steps:
      - id: normalize
        type: preprocessor
        is_train: false       # Load from MLflow
        alias: latest
        instance_key: normalize_model
      - id: cluster
        type: clustering
        wiring:
          inputs:
            model: cluster_model   # Load from MLflow
```

The generator:
1. Detects all artifacts to load (normalize, cluster)
2. Transforms steps to use loaded artifacts
3. Removes training-only configs (hyperparams, log_artifact)
4. Sets correct instance_key and alias
5. Generates evaluators for all model producers

## See Also

- Main documentation: `docs/architecture.md`
- Pipeline configs: `mlproject/configs/pipelines/`
- ConfigGenerator source: `mlproject/src/utils/config_generator.py`
