# Project Structure Overview

This project adopts a production-oriented machine learning architecture with strict boundaries between training, feature preprocessing, and serving/inference.
While a full online/offline feature store is not implemented, the layout is deliberately structured to remain feature-store–ready, enabling seamless integration of systems like Feast, Feathr, or Tecton without significant refactoring.

---

## 1. Why This Structure?

Modern ML systems operate through two distinct—but tightly aligned—execution paths:

### **(1) Training Flow**
- Works with offline datasets, dataloaders, and feature generation pipelines.
- Supports heavy, compute-intensive transformations.
- Must be fully reproducible, driven by versioned configurations.
- Performs model training and evaluation under controlled conditions.

### **(2) Serving Flow**
- Executes only lightweight, deterministic preprocessing followed by model inference.
- Must be fast, stable, and latency-safe for production.
- Consumes online inputs, not training-time dataloader logic or offline transformations.

Separating these flows prevents:
- Preprocessing mismatches and training–serving drift
- Feature leakage from offline logic leaking into production
- Tight coupling between research code and deployment code
- Latency risks from heavy transformations at inference time

This structure keeps the code simple while staying aligned with how real systems integrate with an online/offline Feature Store later.

---

## 2. MLProject Directory Structure Documentation
*Perspective: OOP + Modern MLOps*

---

### Root Package

- `configs/`
  Configuration-driven design separates **hyperparameters, paths, experiment overrides**.
  - `base/` — Reusable core configs.
    - `data.yaml`, `evaluation.yaml`, `mlflow.yaml`, `model.yaml`, `preprocessing.yaml`, `training.yaml`
  - `experiments/` — Specific experiment configs.
    - `etth1.yaml`, `etth2.yaml`, `etth3.yaml`
    > Demonstrates **reproducibility & parameterized experimentation**.

- `data/`
  Raw datasets; **source-of-truth separated**.
  - `ETTh1.csv` — example time-series dataset.

- `run.py`
  CLI / orchestrator for training, evaluation, serving. Implements **single responsibility principle**.

---

### Serving Layer (`serve/`)
Separation of concerns: API, services, schemas, deployment scripts.

- `api.py` — FastAPI endpoints; minimal logic, delegates to service classes.
- `models_service.py` — Encapsulates:
  - Model loading (MLflow or local)
  - Input preprocessing
  - Windowing
  - Prediction
  > **Reusable service class** for API, Ray, or CLI.
- `schemas.py` — Pydantic models for validation.
- `ray/` — Ray Serve deployment.
  - `ray_deploy.py` — PreprocessingService & ModelService, exposes ForecastAPI.

---

### Core ML Library (`src/`)

#### `datamodule/` — Dataset abstraction & modular pipelines
- `dataset.py` — Base dataset interface
- `dm_factory.py` — Factory pattern for data modules
- `tsbase.py` — Base class for time-series datasets
- `tsdl.py` — Deep learning DataLoader
- `tsml.py` — Traditional ML DataLoader

#### `eval/` — Evaluation layer
- `base.py` — Abstract evaluation class
- `regression_eval.py` — MAE, MSE, SMAPE; **strategy pattern** for metric selection
- `ts_eval.py` — Time-series specific evaluation

#### `models/` — Model implementations & wrappers
- `base.py` — Base model interface (`fit`, `predict`)
- `model_factory.py` — Factory pattern
- `nlinear_wrapper.py`, `tft_wrapper.py`, `xgboost_wrapper.py` — Adapter pattern for unified API

#### `pipeline/` — Orchestration layer
- `base.py` — Abstract BasePipeline class
- `config_loader.py` — Hierarchical config loader
- `eval_pipeline.py`, `run_pipeline.py`, `serve_pipeline.py`, `training_pipeline.py` — Full ML lifecycle orchestration

#### `preprocess/` — Feature engineering
- `base.py` — Abstract Preprocessor interface
- `engine.py` — Core transformation logic
- `offline.py` — Training-time preprocessing
- `online.py` — Serving-time preprocessing
> Guarantees **train-serving parity** (critical in MLOps)

#### `tracking/` — MLflow & experiment tracking
- `config_logger.py` — Logging setup
- `experiment_manager.py` — Manages experiments lifecycle
- `mlflow_manager.py` — MLflow wrapper
- `model_logger.py` — Logs model metadata
- `pyfunc_wrapper.py` — Makes arbitrary models MLflow PyFunc compatible
- `registry_manager.py` — MLflow registry interactions
- `run_manager.py` — Run lifecycle & reproducibility metadata
> Supports **reproducibility, auditability, and experiment versioning**

#### `trainer/` — Training abstractions
- `base_trainer.py` — Base trainer interface
- `dl_trainer.py` — Deep learning trainer
- `ml_trainer.py` — Traditional ML trainer
- `trainer_factory.py` — Dynamic trainer instantiation (**factory pattern**)

---

### Design Patterns & Best Practices Highlighted
1. **Encapsulation:** Expose minimal public API, hide internal logic.
2. **Modularity & Reusability:** Services reusable across FastAPI, Ray, CLI.
3. **Factory Pattern:** `ModelFactory`, `TrainerFactory`, `DataModuleFactory`.
4. **Adapter Pattern:** Standardizes `predict` interface across heterogeneous models.
5. **Separation of Concerns:** `preprocess/` vs `models/` vs `pipeline/` vs `tracking/`.
6. **Configuration-Driven Design:** All hyperparameters, paths, experiment settings live in `configs/`.
7. **Reproducibility & Experiment Tracking:** MLflow + `run_manager` ensures audit trail.
8. **Train-Serving Parity:** Offline and online preprocessing ensures consistent production input.
9. **Deployment Ready:** FastAPI + Ray Serve for **scalable, async endpoints**.

---

# Usage Guide

## 1. Environment Setup

Create and initialize the development environment:

```bash
make venv
source mlproject_env/bin/activate
```


## 2. Apply Code Style Fixes

Format and clean the project automatically:
```bash
make style
```

## 3. Run Tests

Run all style, type, and unit tests:

```bash
make test
```
This includes:

- flake8
- mypy
- pytest
- pylint


## 4. Commands — run & serve

Run mlflow server
```bash
mlflow ui --port 5000
```
Run training pipeline

```bash
python -m mlproject.src.pipeline.run_pipeline train --config mlproject/configs/experiments/etth1.yaml
python -m mlproject.src.pipeline.run_pipeline eval --config mlproject/configs/experiments/etth1.yaml
python -m mlproject.src.pipeline.run_pipeline test --config mlproject/configs/experiments/etth1.yaml --input sample_input.csv
```

Run Cross-Validation
```bash
python -m mlproject.src.pipeline.run_pipeline cv --config mlproject/configs/experiments/etth2.yaml

python -m  mlproject.src.run_cv --config mlproject/configs/experiments/etth2.yaml --n-splits 5  --test-size 24
```

Tuning
```bash
python -m mlproject.src.pipeline.run_pipeline tune --config mlproject/configs/experiments/etth3_tuning.yaml
python -m mlproject.src.pipeline.run_pipeline eval --config mlproject/configs/experiments/etth3_tuning.yaml
```


Start FastAPI API
```bash
uvicorn mlproject.serve.api:app --reload
```

Deployment with Ray
```bash
ray start --head
python mlproject/serve/ray/ray_deploy.py
curl http://localhost:8000/health
ray stop
```
