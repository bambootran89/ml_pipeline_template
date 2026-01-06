# Production-Grade ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-2b5b84?logo=python&logoColor=white)](https://www.python.org/)
[![Feast](https://img.shields.io/badge/Feast-Feature%20Store-388e3c?logo=feast&logoColor=white)](https://feast.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0d47a1?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-6a1b9a?logo=optuna&logoColor=white)](https://optuna.org/)
[![Ray Serve](https://img.shields.io/badge/Ray%20Serve-Serving-1565c0?logo=ray&logoColor=white)](https://docs.ray.io/en/latest/serve/index.html)
[![Hydra](https://img.shields.io/badge/Hydra-Config-ef6c00?logo=hydra&logoColor=white)](https://hydra.cc/)
[![Docker](https://img.shields.io/badge/Deployment-Docker%20%7C%20K8s-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

## Overview
This ML platform is designed for **production-ready ML projects**, emphasizing:
- **Eliminating training-serving skew** through unified artifact packaging
- **Feature Store integration** for consistent feature engineering
- **Distributed serving** with Ray for scalability
- **Design patterns** (Strategy, Facade, Factory) for maintainability
- **MLOps best practices** (versioning, monitoring, reproducibility)

## Key Takeaways

### For Data Scientists
- **Focus on modeling**: Preprocessing handled automatically
- **Experiment tracking**: All runs logged to MLflow
- **Easy experimentation**: Change config, not code
- **Reproducibility**: Versioned artifacts + configs

### For MLOps Engineers
- **Zero-skew deployment**: Preprocessor + Model bundled
- **Alias-based rollout**: production/staging/latest
- **Scalable serving**: Ray Serve auto-scaling
- **Monitoring**: Built-in metrics and dashboards

### For Data Engineers
- **Feature Store**: Feast for feature management
- **Consistent features**: Same definitions for training/serving
- **Materialization**: Offline → Online sync
- **Multi-entity**: Batch queries for efficiency

## Documentation
- [Architecture](docs/architecture.md)
- [Directory Structure](docs/directorystructure.md)
- [Offline Workflow](docs/offlineworkflow.md)
- [Online Workflow](docs/onlineworkflow.md)
- [Preprocessing](docs/preprocessing.md)
- [Adding New Model](docs/adding_new_model.md)

---

# Getting Started
## 1. Prerequisites
Python 3.10+

Virtual Environment (recommended)

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/bambootran89/ml_pipeline_template.git
cd ml_pipeline_template

# 1. Create virtual environment using Python 3.10
python3.10 -m venv py3.10

# 2. Activate virtual environment (Linux/macOS)
source py3.10/bin/activate

# 3. Install dependencies
pip install -r requirements/prod.txt
pip install -r requirements/dev.txt   # optional, for testing

# 4. Install project in editable mode
pip install -e .
```

## 3. Quality Assurance
We enforce code quality via pre-commit hooks.
```bash
# Run all tests
make test

# Auto-format code
make style
```

## 4. Quick Start
Start the MLflow server to visualize results:
```bash
mlflow ui --port 5000
```

Run a standard training experiment:
```bash
python -m mlproject.src.pipeline.compat.v1.run train \
  --config mlproject/configs/experiments/etth1.yaml
```

Run a eval experiment.  (alias: latest, production, staging)
```bash
python -m mlproject.src.pipeline.compat.v1.run eval \
  --config mlproject/configs/experiments/etth1.yaml \
  --alias latest
```

Run a serving. (alias: latest, production, staging)
```bash
python -m mlproject.src.pipeline.compat.v1.run serve \
  --config mlproject/configs/experiments/etth1.yaml \
  --input ./sample_input.csv \
  --alias latest
```

---

# DAG-Based Pipeline System (New!)

## Why DAG Pipelines?

The traditional monolithic pipeline approach has limitations when building complex ML workflows:

| Challenge | Monolithic Approach | DAG Pipeline Approach |
|-----------|--------------------|-----------------------|
| **Reusability** | Copy-paste code between projects | Compose reusable steps via YAML |
| **Flexibility** | Hard-coded execution order | Dynamic DAG with dependencies |
| **Experimentation** | Change code to try new flows | Swap pipeline configs only |
| **Complex Workflows** | Difficult to implement | Native support for parallel, branching, nesting |
| **Separation of Concerns** | Config + Logic mixed | Experiment config vs Pipeline structure |

### Key Design Principle: Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT CONFIG                        │
│              (WHAT to train/evaluate)                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Data source (path, type, columns)                │   │
│  │  • Model selection (xgboost, tft, nlinear)          │   │
│  │  • Hyperparameters (learning_rate, n_estimators)    │   │
│  │  • MLflow settings (tracking, registry)             │   │
│  └─────────────────────────────────────────────────────┘   │
│              configs/experiments/etth3.yaml                 │
└─────────────────────────────────────────────────────────────┘
                            +
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE CONFIG                          │
│              (HOW to execute steps)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  • Step definitions (id, type, depends_on)          │   │
│  │  • Data wiring (input/output key mapping)           │   │
│  │  • Execution flow (sequential, parallel, branch)    │   │
│  │  • Advanced patterns (sub-pipelines, conditions)    │   │
│  └─────────────────────────────────────────────────────┘   │
│              configs/pipelines/standard_train.yaml          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    MERGED EXECUTION                         │
│  Same experiment config + Different pipeline configs        │
│  = Different workflows without changing code!               │
└─────────────────────────────────────────────────────────────┘
```

## Available Pipeline Types

### Standard Pipelines
| Pipeline | Description | Use Case |
|----------|-------------|----------|
| `standard_train.yaml` | Training with profiling | Basic model training + output analysis |
| `standard_eval.yaml` | Model evaluation | Test model on new data |
| `standard_serve.yaml` | Inference pipeline | Production predictions |
| `standard_tune.yaml` | Hyperparameter tuning | Optuna optimization |

### Advanced Pipelines
| Pipeline | Description | Use Case |
|----------|-------------|----------|
| `kmeans_then_xgboost.yaml` | Two-stage: Clustering → Classification | Analysis |
| `parallel_ensemble.yaml` | Train multiple models in parallel | Ensemble methods |
| `conditional_branch.yaml` | Select model based on data size | Adaptive model selection |
| `nested_feature_pipeline.yaml` | Sub-pipeline for feature engineering | Modular feature pipelines |

## Basic Usage

### Training
```bash
# Long form, not using feast
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/etth3.yaml \
    --pipeline mlproject/configs/pipelines/standard_train.yaml

# Short form, using feast
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_train.yaml

# dynamic_adapter, Long form, not using feast
python -m mlproject.src.pipeline.dag_run train \
    --experiment mlproject/configs/experiments/tabular.yaml \
    --pipeline mlproject/configs/pipelines/dynamic_adapter_train.yaml
```

### Evaluation
```bash
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_eval.yaml \
    -a latest  # or production, staging

# Generic model
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/tabular.yaml \
    -p mlproject/configs/pipelines/dynamic_adapter_eval.yaml \
    -a latest
```

### Serving (CSV input)
```bash
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_serve.yaml \
    -i ./sample_input.csv \
    -a latest
```

### Serving (Feast Feature Store)
```bash
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3_feast.yaml \
    -p mlproject/configs/pipelines/standard_serve.yaml \
    -a latest \
    --time_point "now"
```

### Hyperparameter Tuning
```bash
python -m mlproject.src.pipeline.dag_run tune \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/standard_tune.yaml \
    -n 50  # number of trials
```

---

## Training with Profiling

Pipeline `standard_trainyaml` includes a **profiling step** that automatically analyzes pipeline outputs after training:

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/tabular.yaml \
    -p mlproject/configs/pipelines/standard_train.yaml
```

### What Profiling Reports

```
============================================================
[profiling] PIPELINE PROFILING REPORT
============================================================

[Metrics Summary]
  evaluate_metrics:
    - mae: 0.0234
    - rmse: 0.0456
    - mape: 2.34

[Cluster Analysis]
  kmeans_labels:
    - n_clusters: 5
    - balance_ratio: 0.72
    - distribution: {0: 1200, 1: 980, 2: 1100, 3: 850, 4: 1050}

[Prediction Statistics]
  inference_predictions:
    - mean: 0.523, std: 0.156
    - range: [0.012, 0.987]

============================================================
```

### Pipeline Flow with Profiling

```
load_data → preprocess → train_model → evaluate → profiling → log_results
                                                       ↓
                                            [Profile Report]
                                            • Metrics summary
                                            • Cluster distributions
                                            • Prediction statistics
```

### Custom Profiling Configuration

```yaml
# In pipeline YAML
- id: "profiling"
  type: "profiling"
  enabled: true
  depends_on: ["evaluate"]
  exclude_keys: ["cfg", "preprocessor", "df", "train_df"]  # Skip these keys
  include_keys: []  # Empty = profile all (except excluded)
```

---

## Auto-Generate Eval/Serve Configs

Automatically generate evaluation and serving configs from your training config:

### Generate Both Configs

```bash
python -m mlproject.src.pipeline.dag_run generate \
    --train-config mlproject/configs/experiments/tabular.yaml \
    --output-dir mlproject/configs/generated \
    --alias latest
```

**Output:**
```
============================================================
[RUN] GENERATING CONFIGS
============================================================

[RUN] Source: mlproject/configs/experiments/tabular.yaml
[RUN] Output: mlproject/configs/generated
[ConfigGenerator] Saved: mlproject/configs/generated/tabular_eval.yaml
[ConfigGenerator] Saved: mlproject/configs/generated/tabular_serve.yaml

Generated configs:
  - Eval:  mlproject/configs/generated/tabular_eval.yaml
  - Serve: mlproject/configs/generated/tabular_serve.yaml
```

### Generate Only Eval Config

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/experiments/tabular.yaml \
    -o mlproject/configs/generated \
    -a latest \
    --type eval
```

### Generate Only Serve Config

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/experiments/tabular.yaml \
    -o mlproject/configs/generated \
    -a staging \
    --type serve
```


## Run Eval with Generated Config

After generating configs, run evaluation:

```bash
# Step 1: Generate configs
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/experiments/tabular.yaml \
    -o mlproject/configs/generated \
    -a latest

# Step 2: Run evaluation with generated config
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/generated/tabular_eval.yaml \
    -a latest
```

### Using Different Aliases

```bash
# Evaluate latest model
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/generated/tabular_eval.yaml \
    -a latest

# Evaluate production model
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/generated/tabular_eval.yaml \
    -a production

# Evaluate staging model
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/generated/tabular_eval.yaml \
    -a staging
```

### Run Serve with Generated Config

```bash
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/generated/tabular_serve.yaml \
    -i ./test_data.csv \
    -a latest
```

## Advanced Pipeline Examples

### 1. Two-Stage Pipeline (KMeans → XGBoost)
Use clustering labels as additional features for classification.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/kmeans_then_xgboost.yaml
```

**Flow:**
```
load_data → preprocess → kmeans_features → xgboost_model → evaluate → log
                              ↓
                    [cluster_labels]
```

### 2. Parallel Ensemble
Train multiple models simultaneously and evaluate each.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/parallel_ensemble.yaml
```

**Flow:**
```
load_data → preprocess → ┬─ xgboost_branch -─┬→ eval_xgb ─┬→ log
                         ├─ catboost_branch ─┤→ eval_cat ─┤
                         └─ kmeans_branch ───┘            │
                              (parallel)                  ↓
```

### 3. Conditional Branching
Automatically select model based on dataset size.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/conditional_branch.yaml
```

**Flow:**
```
load_data → preprocess → branch ─┬─ [data_size > 100] → TFT (deep learning)
                                 └─ [data_size ≤ 100] → XGBoost (ML)
                                          ↓
                                      evaluate → log
```

### 4. Nested Sub-Pipeline
Encapsulate feature engineering as a reusable sub-pipeline.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/nested_suppipeline.yaml
```

**Flow:**
```
load_data → feature_pipeline (sub) → train_model → evaluate → log
                   ↓
           ┌──────────────┐
           │  normalize   │
           │      ↓       │
           │   cluster    │
           │ (features)   │
           └──────────────┘
```

### 5. Nested Sub-Pipeline (Variant)
Alternative sub-pipeline configuration.

```bash
python -m mlproject.src.pipeline.dag_run train \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/pipelines/nested_suppipeline.yaml
```

## Data Wiring System

The DAG pipeline supports flexible data routing between steps via `wiring` configuration:

```yaml
- id: "train_model"
  type: "trainer"
  depends_on: ["preprocess", "clustering"]
  wiring:
    inputs:
      data: "preprocessed_data"      # Read from context["preprocessed_data"]
      features: "cluster_labels"     # Read from context["cluster_labels"]
    outputs:
      model: "final_model"           # Write to context["final_model"]
      datamodule: "final_dm"         # Write to context["final_dm"]
```

This enables:
- **Custom key mapping**: Override default input/output keys
- **Multi-input steps**: Combine outputs from multiple upstream steps

## Step Types Reference

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `data_loader` | Load data from CSV/Feast | - |
| `preprocessor` | Fit/transform features | `is_train`, `alias` |
| `trainer` | Train model | `output_as_feature`, `use_tuned_params` |
| `evaluator` | Compute metrics | `model_step_id` |
| `logger` | Log to MLflow | `model_step_id`, `eval_step_id` |
| `mlflow_loader` | Load from MLflow Registry | `alias` |
| `inference` | Generate predictions | `model_step_id`, `save_path` |
| `tuner` | Optuna hyperparameter search | `n_trials` |
| `profiling` | Analyze pipeline outputs | `include_keys`, `exclude_keys` |
| `parallel` | Execute branches concurrently | `branches`, `max_workers` |
| `branch` | Conditional execution | `condition`, `if_true`, `if_false` |
| `sub_pipeline` | Nested pipeline | `pipeline`, `output_prefix` |
| `clustering` | Clustering with auto-feature output | `model_name`, `output_as_feature` |

---

# Feature Store Integration

Create and populate the Feast feature store.
Run scheduled periodic data ingestion (e.g., daily)
```bash
# Ingest Batch: Typically runs in large-scale data processing jobs (Spark/Pandas). It may run for a long time.
python -m mlproject.src.pipeline.feature_ops.ingest_batch_etth1 \
    --csv mlproject/data/ETTh1.csv \
    --repo feature_repo_etth1

python -m mlproject.src.pipeline.feature_ops.ingest_titanic \
    --csv mlproject/data/titanic.csv \
    --repo titanic_repo
```

Sync data to the online store for API inference
```bash
# Materialize: Can run independently. Example: you can update the online store
# hourly without rerunning feature engineering if offline features are already available.
python -m mlproject.src.pipeline.feature_ops.materialize_etth1 \
    --repo feature_repo_etth1 \
    --data feature_repo_etth1/data/features.parquet

python -m mlproject.src.pipeline.feature_ops.materialize_titanic \
    --repo titanic_repo \
    --data titanic_repo/data/titanic.parquet
```

Training with automatic data loading from Feast:
```bash
python -m mlproject.src.pipeline.compat.v1.run train \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

Evaluation with automatic data loading from Feast:
```bash
python -m mlproject.src.pipeline.compat.v1.run eval \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

Run the serving stage with automatic feature loading from Feast
```bash
python -m mlproject.src.pipeline.compat.v1.run serve \
    --config mlproject/configs/experiments/etth1_feast.yaml
```

---

# Workflows & Capabilities

## 1. Cross-Validation (Backtesting)
Validates model stability across time folds.
```bash
python -m mlproject.src.pipeline.compat.v1.run cv \
  --config mlproject/configs/experiments/etth2.yaml
```

## 2. Hyperparameter Tuning (Optuna)
Runs Bayesian Optimization to find best parameters, then auto-retrains the best model.
```bash
python -m mlproject.src.pipeline.compat.v1.run tune \
  --config mlproject/configs/experiments/etth1_tuning.yaml


python -m mlproject.src.pipeline.compat.v1.run tune \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

## 3. Serving (Inference)
Deploys the model using FastAPI.
```bash
# Start FastAPI
CONFIG_PATH=mlproject/configs/experiments/etth1_feast.yaml uvicorn mlproject.serve.api:app --reload
```
Deploys the model using Ray Serve.
```bash
# OR Start Ray Serve
CONFIG_PATH=mlproject/configs/experiments/etth1_feast.yaml python mlproject/serve/ray/ray_deploy.py
```
Test API services: mlproject/configs/experiments/etth1.yaml
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json"
  -d '{
    "data": {
      "HUFL": [-0.15, 0.08, 0.01, -0.01, 0.21, -0.15, 0.12, 0.05, -0.08, 0.18, -0.12, 0.22, 0.03, -0.18, 0.15, -0.05, 0.28, -0.22, 0.08, -0.15, 0.32, -0.28, 0.12, -0.18],
      "MUFL": [1.14, 1.06, 0.93, 1.11, 0.96, 1.05, 0.98, 1.12, 0.95, 1.08, 0.92, 1.15, 0.88, 1.22, 0.85, 1.18, 0.82, 1.25, 0.78, 1.32, 0.75, 1.38, 0.72, 1.42],
      "mobility_inflow": [1.24, 4.42, 7.28, 1.03, 0.73, 2.5, 3.2, 4.1, 1.8, 5.3, 2.1, 6.4, 1.5, 7.8, 3.6, 4.9, 2.7, 8.2, 1.9, 5.5, 3.8, 6.7, 2.3, 4.2]
    }
  }'
```

Test API services: mlproject/configs/experiments/etth1etth1_feast.yaml
```bash
curl -X POST http://localhost:8000/predict/feast/batch \
  -H "Content-Type: application/json" \
  -d '{"time_point":"now","entities":[1,2,3,4,5],"entity_key":"location_id"}'

curl -X POST http://localhost:8000/predict/feast \
  -H "Content-Type: application/json" \
  -d '{"time_point":"now","entities":[1,2,3,4,5],"entity_key":"location_id"}'
```

---

# Docker & Kubernetes

Build Docker Image
```bash
docker build -t ml-pipeline:latest .
```

Run Locally with Docker
```bash
docker run -p 8000:8000 ml-pipeline:latest
```

Deploy to Kubernetes
```bash
# Apply Training Job
kubectl apply -f k8s/job-training.yaml

# Apply API Service
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/service-api.yaml
```
