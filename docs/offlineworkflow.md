# Offline Workflow (Training Pipeline)

## Overview

The training pipeline orchestrates model development with emphasis on:
- **Anti-leakage**: Preprocessing fitted only on training folds
- **Reproducibility**: All experiments tracked in MLflow
- **Hyperparameter tuning**: Optuna integration
- **Cross-validation**: Time-series aware splitting
- **Artifact bundling**: Preprocessor + Model packaged together

---

## Complete Training Flow

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 25,
    "rankSpacing": 30,
    "curve": "linear",
    "padding": 6
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "15px",
    "fontWeight": "600",
    "lineColor": "#2196f3",
    "edgeStrokeWidth": "2.2px",
    "nodeBorderRadius": "10px",
    "clusterBorderRadius": "12px"
  }
}}%%

flowchart TD
    %% Node styles
    classDef input fill:#fff9c4,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600
    classDef orchestrator fill:#e3f2fd,stroke:#1565c0,stroke-width:2.2px,color:#000,font-weight:600
    classDef compute fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.2px,color:#000,font-weight:600
    classDef package fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600
    classDef storage fill:#263238,stroke:#000,stroke-width:2.5px,color:#fff,font-weight:600
    classDef hotpath stroke:#2196f3,stroke-width:3.2px
    classDef parampath stroke:#0277bd,stroke-width:2.2px,stroke-dasharray:6 6
    classDef note fill:#fff3e0,stroke:#ef6c00,stroke-width:1.5px,color:#000,font-style:italic,font-size:14px

    %% Phases
    subgraph Phase1["1) Config & Data"]
        direction TB
        Config["Hydra Config"]:::input
        DataLoader["DataIO Loader"]:::input
        Feast["Feast Store"]:::input
    end

    subgraph Phase2["2) Time-Series Safe Split"]
        direction TB
        CVManager["CV Manager"]:::orchestrator
        Split["Train / Val Index"]:::orchestrator
    end

    subgraph Phase3["3) Feature Transform (Per Fold)"]
        direction LR
        TransFit["fit() on TRAIN"]:::compute
        TransTrain["transform() TRAIN"]:::compute
        TransVal["transform() VAL"]:::compute
    end

    subgraph Phase4["4) Model Train & Eval"]
        direction TB
        Optuna["Optuna Tuning (Opt)"]:::orchestrator
        Trainer["Trainer.fit()"]:::compute
        Evaluate["Validation Metrics"]:::compute
    end

    subgraph Phase5["5) Serialize & Wrap"]
        direction LR
        Serializer["Save Preprocess + Model"]:::package
        Wrapper["PyFuncWrapper"]:::package
    end

    subgraph Phase6["6) MLflow Logging"]
        direction TB
        MLflow["Registry + Alias"]:::storage
    end

    %% Main flow
    Config & DataLoader ==> Feast:::hotpath
    Feast ==> CVManager:::hotpath
    CVManager ==> Split:::hotpath
    Split -->|"Train idx"| TransFit:::hotpath
    Split -->|"Val idx"| TransVal:::hotpath
    TransFit ==> TransTrain:::hotpath
    TransFit ==> TransVal:::hotpath
    TransTrain ==> Trainer:::hotpath
    TransVal ==> Evaluate:::hotpath
    Trainer ==> Evaluate:::hotpath
    TransFit --> Serializer:::hotpath
    Trainer --> Serializer:::hotpath
    Serializer ==> Wrapper:::hotpath
    Wrapper ==> MLflow:::hotpath

    %% Parameter tuning flow
    Optuna -.->|"Suggest HP"| Trainer:::parampath
    Evaluate -.->|"Score"| Optuna:::parampath

    %% Notes
    N1["No Leakage:<br/>Stats only from Train fold"]:::note
    N2["Self-Contained:<br/>Preprocess + Model"]:::note
    N3["Loop:<br/>Repeat for each fold"]:::note
    N4["Best Model:<br/>Retrain on full data"]:::note

    N1 -.-> TransFit
    N2 -.-> Wrapper
    N3 -.-> CVManager
    N4 -.-> MLflow

    %% Global edge style
    linkStyle default stroke:#2196f3,stroke-width:2.2px

```

---

## Phase 1: Configuration & Data Loading

**Key Config Parameters:**
```yaml
# etth1_feast.yaml
data:
  path: "feast://weather_repo"  # Feast URI
  type: "timeseries"
  featureview: "hourly_features"
  features: ["HUFL", "MUFL", "mobility_inflow"]
  target_columns: ["HUFL"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  entity_key: "location_id"
  entity_id: 1

experiment:
  model: "xgboost"
  hyperparams:
    n_estimators: 100
    learning_rate: 0.1
```

---

## Phase 2: Cross-Validation (Time-Series Aware)

## Phase 3: Feature Engineering (Per Fold)

## Phase 4: Model Training with Optuna

**Optuna Config:**
```yaml
tuning:
  n_trials: 30
  n_splits: 3
  optimize_metric: "mae_mean"
  direction: "minimize"

  # Parameter search space
  xgboost:
    n_estimators:
      type: "int"
      range: [50, 300]
    learning_rate:
      type: "float"
      range: [0.01, 0.1]
      log: true
    max_depth:
      type: "int"
      range: [3, 10]
```

## Phase 5: Artifact Packaging

## Phase 6: MLflow Logging

---

## Summary: Complete Pipeline Flow

**Command:**
```bash
python -m mlproject.src.pipeline.run train \
  --config mlproject/configs/experiments/etth1_feast.yaml
```

**Steps:**
1. Load config via Hydra
2. Load data from Feast (historical)
3. Split into CV folds (time-series aware)
4. For each fold:
   - Fit preprocessor on TRAIN
   - Transform TRAIN and VAL
   - Train model with suggested hyperparams
   - Evaluate on VAL
   - Report metrics to Optuna
5. Select best hyperparameters
6. Retrain on full dataset
7. Package preprocessor + model
8. Log to MLflow Registry
9. Assign alias (latest/staging/production)

**Output:**
- MLflow Run with all metrics, params, artifacts
- Registered model version ready for deployment
- Reproducible artifact bundle

**Key Guarantees:**
- **No data leakage**: Val never seen during preprocessing fit
- **Bundled artifacts**: Preprocessor + Model together
- **Reproducible**: Config + Version → Exact same result
- **Tracked**: All experiments logged to MLflow
