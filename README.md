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

## 2. Directory Reference
```text
mlproject/
├── run.py                # Entry for training
├── serve/
│ └── api.py              # Inference/serving API
├── configs/
│ ├── base/.yaml          # Modular config blocks
│ └── experiments/.yaml   # Experiment-level configs
└── src/
  ├── data/               # Dataset + Dataloader (training only)
  ├── preprocess/         # Offline = heavy, Online = serving-safe
  ├── models/             # Model registry + wrappers
  ├── trainer/            # Training loop
  ├── eval/               # Evaluation logic
  ├── config_loader.py    # Unified config loader
  └── run_pipeline.py     # Orchestrates training pipeline
```

---

## 3. Key Design Principles

### **Clear Separation of Concerns**
- `src/data/` is dedicated **exclusively to training**, handling datasets and dataloaders.
- `src/preprocess/offline.py` performs the full offline preprocessing pipeline, including:
  + heavy feature engineering
  + fitting scalers/statistics
  + generating covariates
  + saving preprocessing artifacts
  + executing all steps based on versioned configurations

This is the heavyweight, reproducible pipeline used for model training and evaluation.

- `src/preprocess/online.py` is restricted to serving-time, transform-only preprocessing, containing only lightweight and deterministic steps.
- `src/preprocess/engine.py` acts as the runtime orchestrator:

  + a singleton that loads artifacts exactly once
  + enforces schema consistency
  + guarantees low-latency preprocessing for inference

This strict separation ensures production safety and prevents accidental coupling with training logic.

### **Config-Driven Architecture**
- All components—data, preprocessing, models, evaluation—are fully modular and configured through versioned YAML files.
- This enables reproducible experiments, controlled variations, and seamless component swapping without code changes.

### **Feature-Store-Ready Design**
Even in the absence of a formal feature store, the project mirrors its conceptual structure:
- Offline preprocessing aligns with what an **offline feature store** would compute (batch, historical, heavy transforms).
- Online preprocessing reflects what an **online feature store** would provide (real-time, latency-safe features).
- This makes future integration with Feast, Feathr, Tecton, or similar systems effectively plug-and-play.


---

## 4. Summary

This architecture strikes a balance between **simplicity** and **production scalability**:

- Lean, maintainable codebase
- Strong separation between training and serving paths
- Guaranteed consistency with no preprocessing drift
- Designed for seamless integration with feature stores, model registries, and monitoring systems

Overall, it provides a lightweight yet production-aligned template for building reliable, end-to-end ML pipelines.


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
Run training pipeline

```bash
python mlproject/run.py --config mlproject/configs/experiments/etth1.yaml
```
Start API
```bash
uvicorn mlproject.serve.api:app --reload
```
