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
    "nodeSpacing": 70,
    "rankSpacing": 90,
    "curve": "linear",
    "padding": 15
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "16px",
    "fontWeight": "500",
    "lineColor": "#222",
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
    classDef hotpath stroke:#c62828,stroke-width:3.5px
    classDef parampath stroke:#0277bd,stroke-width:2.2px,stroke-dasharray:6 6
    classDef note fill:#fff3e0,stroke:#ef6c00,stroke-width:1.5px,color:#000,font-style:italic,font-size:15px

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
    N1["No Leakage:\nStats only from Train fold"]:::note
    N2["Self-Contained:\nPreprocess + Model"]:::note
    N3["Loop:\nRepeat for each fold"]:::note
    N4["Best Model:\nRetrain on full data"]:::note

    N1 -.-> TransFit
    N2 -.-> Wrapper
    N3 -.-> CVManager
    N4 -.-> MLflow
```

---

## Phase 1: Configuration & Data Loading

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":80,"rankSpacing":110,"curve":"linear"},
  "themeVariables": {"primaryColor":"#e3f2fd","lineColor":"#111","edgeStrokeWidth":"2.4px","fontSize":"16px","fontWeight":"600","fontFamily":"Inter, Arial"}
}}%%

flowchart LR

    %% Stronger Styles
    classDef config fill:#fff9c4,stroke:#f9a825,stroke-width:2.6px,color:#000,font-weight:700
    classDef loader fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.6px,color:#000,font-weight:700
    classDef data fill:#e3f2fd,stroke:#1565c0,stroke-width:2.6px,color:#000,font-weight:700
    classDef hotpath stroke:#c62828,stroke-width:4px
    classDef edgeMain stroke-width:3px,color:#000,font-weight:700

    %% Config Section
    subgraph ConfigSources["Config"]
        BaseConfig["Base Files"]:::config
        ExpConfig["Experiment"]:::config
    end

    Hydra["Hydra Merge"]:::config
    Factory["Loader Factory"]:::loader

    %% Loaders
    FeastLoader["Feast"]:::loader
    FileLoader["Files"]:::loader

    %% Output Data
    subgraph Data["Data"]
        DataFrame["DataFrame"]:::data
    end

    %% Main Pipeline Flow
    BaseConfig ==> Hydra:::hotpath
    ExpConfig ==> Hydra:::hotpath
    Hydra ==> Factory:::hotpath

    Factory -->|"feast://"| FeastLoader:::hotpath
    Factory -->|"csv/parquet"| FileLoader:::hotpath

    FeastLoader ==>|"historical()"| DataFrame:::hotpath
    FileLoader ==>|"read()"| DataFrame:::hotpath
```

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

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":90,"rankSpacing":130,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","lineColor":"#0a0a0a","edgeStrokeWidth":"3px","fontSize":"17px","fontWeight":"650","fontFamily":"Inter, Arial, sans-serif","clusterBorderRadius":"10px","nodeBorderRadius":"8px"}
}}%%

flowchart LR

    %% Stronger Styles (high contrast & clean)
    classDef config fill:#fff9c4,stroke:#f57f17,stroke-width:2.8px,color:#000,font-weight:700
    classDef loader fill:#e8f5e9,stroke:#1b5e20,stroke-width:2.8px,color:#000,font-weight:700
    classDef data fill:#e3f2fd,stroke:#0d47a1,stroke-width:2.8px,color:#000,font-weight:700
    classDef hotpath stroke:#b71c1c,stroke-width:4.2px
    classDef edgeMain stroke-width:3.2px,color:#000,font-weight:700

    %% Config Section
    subgraph ConfigSources["CONFIG INPUTS"]
        direction TB
        BaseConfig["Base Files"]:::config
        ExpConfig["Experiment"]:::config
    end

    Hydra["Hydra Merge"]:::config
    Factory["Loader Factory"]:::loader

    %% Loaders
    FeastLoader["Feast"]:::loader
    FileLoader["Files"]:::loader

    %% Output Data
    subgraph Data["LOADED DATA"]
        direction TB
        DataFrame["DataFrame"]:::data
    end

    %% Main Pipeline Flow (bold priority path)
    BaseConfig ==> Hydra:::hotpath
    ExpConfig ==> Hydra:::hotpath
    Hydra ==> Factory:::hotpath

    Factory ==>|"feast://"| FeastLoader:::hotpath
    Factory ==>|"csv/parquet"| FileLoader:::hotpath

    FeastLoader ==> DataFrame:::hotpath
    FileLoader ==> DataFrame:::hotpath

    %% Edge Labels Styling (optional emphasis)
    linkStyle default stroke:#0a0a0a,stroke-width:3.2px
```

## Phase 3: Feature Engineering (Per Fold)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":60,"rankSpacing":80,"curve":"basis"},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","fontFamily":"Inter, Arial, sans-serif","fontSize":"14px"}
}}%%

flowchart TB
    %% Styles
    classDef data fill:#fff9c4,stroke:#fbc02d,stroke-width:2px
    classDef fit fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
    classDef transform fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    classDef stats fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    classDef warning fill:#ffebee,stroke:#c62828,stroke-width:2px

    %% Raw Data
    TrainRaw["Train Data (Raw)<br/>100 days"]:::data
    ValRaw["Val Data (Raw)<br/>30 days"]:::data

    %% Fit Phase
    subgraph FitPhase["Fit Phase (TRAIN ONLY)"]
        Fit["TransformManager.fit(train_df)<br/>Learn statistics"]:::fit

        subgraph Stats["Learned Statistics"]
            Mean["Mean: [25.5, 60.2, ...]"]:::stats
            Std["Std: [5.1, 10.3, ...]"]:::stats
            Encoders["Label Encoders<br/>{cat_A: 0, cat_B: 1}"]:::stats
        end
    end

    %% Transform Phase
    subgraph TransformPhase["Transform Phase"]
        TransTrain["TransformManager.transform(train_df)<br/>Apply learned stats"]:::transform
        TransVal["TransformManager.transform(val_df)<br/>Apply SAME stats"]:::transform
    end

    %% Transformed Data
    TrainClean["Train Data (Transformed)<br/>Scaled, encoded, imputed"]:::data
    ValClean["Val Data (Transformed)<br/>Using TRAIN statistics"]:::data

    %% Flow
    TrainRaw --> Fit
    Fit --> Mean
    Fit --> Std
    Fit --> Encoders

    Mean --> TransTrain
    Std --> TransTrain
    Encoders --> TransTrain

    Mean --> TransVal
    Std --> TransVal
    Encoders --> TransVal

    TrainRaw --> TransTrain
    ValRaw --> TransVal

    TransTrain --> TrainClean
    TransVal --> ValClean

    %% Warning
    Warning["CRITICAL:<br/>Val data NEVER used in fit()<br/>Prevents data leakage"]:::warning
    Warning -.-> Fit
```

**Code Example:**
```python
# Fit preprocessor on TRAIN fold only
transform_manager = TransformManager(config)
transform_manager.fit(train_df)  # Learn mean, std, etc.

# Transform BOTH train and val using learned statistics
train_transformed = transform_manager.transform(train_df)
val_transformed = transform_manager.transform(val_df)

# Val data uses TRAIN statistics (anti-leakage!)
# Example: If TRAIN mean=25.5, VAL also scales by 25.5
```

---

## Phase 4: Model Training with Optuna

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":110,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","lineColor":"#000","edgeStrokeWidth":"3px","clusterBorderRadius":"10px","nodeBorderRadius":"8px","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"600"}
}}%%

flowchart TB
    %% Styles
    classDef optuna fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef training fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef eval fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef best fill:#e1bee7,stroke:#000,stroke-width:4.2px,color:#000,font-weight:800
    classDef loopEdge stroke:#b71c1c,stroke-width:4.5px

    %% Optuna Setup
    subgraph OptunaSetup["Optuna Hyperparameter Tuning"]
        Study["Optuna Study<br/>optimize_metric: MAE<br/>direction: minimize<br/>n_trials: 30"]:::optuna
    end

    %% Trial Loop
    subgraph TrialLoop["Trial Loop (30 iterations)"]
        Suggest["Suggest Hyperparams<br/>learning_rate: [0.01, 0.1]<br/>n_estimators: [50, 300]<br/>max_depth: [3, 10]"]:::optuna

        subgraph Training["Training"]
            Build["Build Model<br/>with suggested params"]:::training
            Fit["Trainer.fit(train, val)"]:::training
        end

        subgraph Evaluation["Evaluation"]
            Predict["Predict on Val"]:::eval
            Metrics["Compute MAE, RMSE"]:::eval
            Report["Report to Optuna"]:::eval
        end
    end

    %% Best Model
    subgraph BestModel["Best Model Selection"]
        SelectBest["Select best trial<br/>lowest MAE"]:::best
        Retrain["Retrain on full data<br/>(Train + Val)"]:::best
        Final["Final Model<br/>Ready for deployment"]:::best
    end

    %% Flow
    Study ==> Suggest
    Suggest ==> Build
    Build ==> Fit
    Fit ==> Predict
    Predict ==> Metrics
    Metrics ==> Report

    Report ==> Suggest:::loopEdge
    Report ==> SelectBest

    SelectBest ==> Retrain
    Retrain ==> Final

    %% Annotations
    Note1["Loop 30 times<br/>Bayesian optimization"]
    Note2["Best params found<br/>Retrain on all data"]

    Note1 -.-> TrialLoop
    Note2 -.-> BestModel
```

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

---

## Phase 5: Artifact Packaging

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":95,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","lineColor":"#000","edgeStrokeWidth":"3px","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"600","clusterBorderRadius":"10px","nodeBorderRadius":"8px"}
}}%%

flowchart LR
    %% Styles
    classDef component fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef serialized fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef wrapper fill:#e1bee7,stroke:#000,stroke-width:4.2px,color:#000,font-weight:800
    classDef mlflow fill:#37474f,stroke:#000,stroke-width:3.5px,color:#fff,font-weight:800
    classDef noteBox fill:#ffebee,stroke:#000,stroke-width:2px,color:#000,font-size:14px,font-style:italic

    %% Components
    subgraph Components["Trained Components"]
        direction TB
        Preprocessor["TransformManager<br/>- Fitted scalers<br/>- Fitted encoders<br/>- Fitted imputers"]:::component
        Model["Trained Model<br/>- XGBoost/TFT/etc.<br/>- Learned weights<br/>- Hyperparameters"]:::component
    end

    %% Serialization
    subgraph Serialization["Serialization"]
        direction TB
        PrepPkl["preprocessor.pkl<br/>pickle.dump()"]:::serialized
        ModelPkl["model.pkl<br/>or model.pt<br/>torch.save()"]:::serialized
    end

    %% Wrapper
    subgraph Wrapper["PyFunc Wrapper"]
        direction TB
        PyFunc["PyFuncWrapper<br/>__init__(preprocessor, model)<br/><br/>predict(raw_input):<br/>  1. preprocess(raw_input)<br/>  2. model.predict(transformed)<br/>  3. return predictions"]:::wrapper
    end

    %% MLflow
    MLflow["MLflow Artifact<br/>Single versioned unit<br/>Version: 1.2.3<br/>Alias: production"]:::mlflow

    %% Flow
    Preprocessor ==> PrepPkl
    Model ==> ModelPkl

    PrepPkl ==> PyFunc
    ModelPkl ==> PyFunc

    PyFunc ==> MLflow

    %% Benefits
    Note1["Atomic deployment<br/>Zero skew<br/>Reproducible"]:::noteBox
    Note1 -.-> MLflow
```

**Code Example:**
```python
# After training
preprocessor = transform_manager  # Fitted
model = trainer.model  # Trained

# Serialize
import pickle
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Wrap as PyFunc
wrapper = PyFuncWrapper(
    preprocessor=preprocessor,
    model=model
)

# Log to MLflow
mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=wrapper,
    registered_model_name="xgboost_forecaster"
)

# Assign alias
mlflow.set_registered_model_alias(
    name="xgboost_forecaster",
    alias="production",
    version=3
)
```

---

## Phase 6: MLflow Logging

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":115,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","lineColor":"#000","edgeStrokeWidth":"3px","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"600","clusterBorderRadius":"10px","nodeBorderRadius":"8px"}
}}%%

flowchart TB
    %% Styles
    classDef experiment fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef run fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef artifact fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef registry fill:#e1bee7,stroke:#000,stroke-width:3.2px,color:#000,font-weight:800
    classDef note fill:#ffebee,stroke:#000,stroke-width:2px,color:#000,font-style:italic,font-size:14px

    %% Experiment
    Exp["MLflow Experiment<br/>Name: 'xgboost_tuning'"]:::experiment

    %% Run
    subgraph Run["MLFLOW RUN"]
        direction TB
        RunID["Run ID: abc123<br/>Timestamp: 2024-01-15"]:::run

        subgraph Logged["LOGGED ITEMS"]
            direction TB
            Params["Parameters<br/>- n_estimators: 100<br/>- learning_rate: 0.1<br/>- max_depth: 6"]:::artifact
            Metrics["Metrics<br/>- train_mae: 0.15<br/>- val_mae: 0.18<br/>- val_rmse: 0.23"]:::artifact
            Artifacts["Artifacts<br/>- model/<br/>  - preprocessor.pkl<br/>  - model.pkl<br/>  - MLmodel"]:::artifact
            Tags["Tags<br/>- algorithm: xgboost<br/>- dataset: etth1<br/>- best_trial: True"]:::artifact
        end
    end

    %% Registry
    subgraph Registry["MODEL REGISTRY"]
        direction TB
        RegModel["Registered Model<br/>xgboost_forecaster"]:::registry
        Version["Version 3<br/>From run: abc123"]:::registry
        Alias["Alias: production"]:::registry
    end

    %% Flow
    Exp ==> RunID:::experiment
    RunID ==> Params:::hotpath
    RunID ==> Metrics:::hotpath
    RunID ==> Artifacts:::hotpath
    RunID ==> Tags:::hotpath

    Artifacts ==> RegModel:::hotpath
    RegModel ==> Version:::hotpath
    Version ==> Alias:::hotpath
```

**MLflow UI View:**
```
Experiments
├── xgboost_tuning
│   ├── Run abc123 (Latest)
│   │   ├── Parameters: {n_estimators: 100, learning_rate: 0.1, ...}
│   │   ├── Metrics: {val_mae: 0.18, val_rmse: 0.23}
│   │   └── Artifacts: model/ (preprocessor.pkl + model.pkl)
│   └── Run def456
│       └── ...
│
Models
└── xgboost_forecaster
    ├── Version 3 (production) ← Currently serving
    ├── Version 2 (staging)
    └── Version 1
```

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
