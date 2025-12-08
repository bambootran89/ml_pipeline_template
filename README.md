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

## 2. Highlights Inside `src/`

---

- **datamodule/**
  - Provides a unified interface for ML & DL data handling.
  - ML vs DL separated: `tsml.py` for XGBoost/sklearn; `tsdl.py` for PyTorch.
  - `dm_factory.py` allows easy instantiation based on model type.

- **preprocess/**
  - Offline preprocessing (`offline.py`) for training: scaling, feature engineering, missing values.
  - Online preprocessing (`online.py`) for serving: deterministic, ensures no train-serving drift.
  - `engine.py` centralizes all transform logic.

- **models/**
  - Wrap models with consistent API (`fit/predict/save/load`).
  - Supports multiple architectures (`NLinear`, `TFT`) via wrapper classes.

- **trainer/**
  - Centralized training loop for both ML and DL models.
  - TrainerFactory instantiates correct trainer based on model type.
  - DL-specific logic isolated in `dltrainer.py`.

- **eval/**
  - Centralized evaluation metrics (MAE, MSE, SMAPE) for both ML and DL.

- **pipeline/**
  - `run_pipeline.py` orchestrates complete pipeline: config load → datamodule → model → trainer → evaluation → artifact save.
  - Separate pipelines for training, evaluation, and testing (`training_pipeline.py`, `eval_pipeline.py`, `test_pipeline.py`).


---

## DataModule Abstraction

Introduces a clean separation between ML and DL data handling.

### MLDataModule
- Returns numpy arrays for traditional ML models (XGBoost, sklearn).
- Handles train/val/test splits and optional feature selection.

### DLDataModule
- Returns PyTorch DataLoaders with windowed datasets for sequential DL models.
- Encapsulates batch/window creation and ensures temporal splits are correct.

> All DataModule logic lives in `src/data/` and replaces old ad-hoc DataLoader usage.

---

## Clear Separation of Concerns
- `src/data/` is dedicated exclusively to training, now via DataModules.
- `src/preprocess/offline.py` performs heavy, offline feature engineering and artifact saving.
- `src/preprocess/online.py` performs serving-safe, deterministic preprocessing only.
- `src/preprocess/engine.py` orchestrates runtime preprocessing with low-latency guarantees.

> This strict separation ensures production safety and prevents accidental coupling with training logic.

---

## Config-Driven Architecture
- All components—data, preprocessing, models, evaluation—are fully modular and configured through versioned YAML files.
- Enables reproducible experiments, controlled variations, and seamless component swapping without code changes.

---

## Feature-Store-Ready Design
Even in the absence of a formal feature store, the project mirrors its conceptual structure:

- Offline preprocessing aligns with what an **offline feature store** would compute (batch, historical, heavy transforms).
- Online preprocessing reflects what an **online feature store** would provide (real-time, latency-safe features).

> This makes future integration with Feast, Feathr, Tecton, or similar systems effectively plug-and-play.


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
python -m mlproject.src.pipeline.run_pipeline train --config mlproject/configs/experiments/etth1.yaml
python -m mlproject.src.pipeline.run_pipeline eval --config mlproject/configs/experiments/etth1.yaml
python -m mlproject.src.pipeline.run_pipeline test --config mlproject/configs/experiments/etth1.yaml --input sample_input.csv
```
Start API
```bash
uvicorn mlproject.serve.api:app --reload
```
