# Architecture & Design Philosophy

## Overview

This ML platform is engineered for **production-grade time series forecasting** with emphasis on:
- **Eliminating training-serving skew** through unified artifact packaging
- **Feature Store integration** for consistent feature engineering
- **Distributed serving** with Ray for scalability
- **Design patterns** (Strategy, Facade, Factory) for maintainability
- **MLOps best practices** (versioning, monitoring, reproducibility)

---

## 1. System Architecture Overview

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":40,"rankSpacing":50,"curve":"linear","padding":10},
  "themeVariables": {
    "primaryColor":"#e3f2fd",
    "lineColor":"#2196f3",
    "edgeLabelBackground":"#fff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"14px",
    "fontWeight":"600",
    "clusterBorderRadius":"10px",
    "nodeBorderRadius":"6px"
  }
}}%%

flowchart TB
    %% Styles
    classDef dataSource fill:#fff9c4,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef featureStore fill:#c8e6c9,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef training fill:#bbdefb,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef registry fill:#e1bee7,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef serving fill:#ffccbc,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700
    classDef monitoring fill:#b0bec5,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:700

    %% Data Layer
    subgraph DataLayer["Data Layer"]
        RawData["Raw Data<br/>(CSV/DB/Streaming)"]:::dataSource
        Feast["Feast Feature Store<br/>Offline + Online"]:::featureStore
    end

    %% Training Layer
    subgraph TrainingLayer["Training Layer"]
        Pipeline["Training Pipeline<br/>- CV/Tuning<br/>- Preprocessing<br/>- Model Training"]:::training
        Artifacts["Artifacts<br/>Preprocessor + Model<br/>(Bundled)"]:::training
    end

    %% Registry Layer
    subgraph RegistryLayer["Model Registry"]
        MLflow["MLflow Registry<br/>production/staging/latest"]:::registry
    end

    %% Serving Layer
    subgraph ServingLayer["Serving Layer"]
        FastAPI["FastAPI<br/>(Dev/Small Scale)"]:::serving
        RayServe["Ray Serve<br/>(Production Scale)"]:::serving
    end

    %% Monitoring Layer
    subgraph MonitoringLayer["Monitoring"]
        Metrics["Metrics<br/>Latency/Throughput"]:::monitoring
        Dashboard["Ray Dashboard<br/>MLflow UI"]:::monitoring
    end

    %% Connections
    RawData ==> Feast
    Feast ==> Pipeline
    Pipeline ==> Artifacts
    Artifacts ==> MLflow

    MLflow ==> FastAPI
    MLflow ==> RayServe

    Feast ==> FastAPI
    Feast ==> RayServe

    FastAPI ==> Metrics
    RayServe ==> Metrics
    Metrics ==> Dashboard

    %% Edge emphasis
    linkStyle default stroke:#2196f3,stroke-width:3px
```

---

## 2. Unified Artifact Packaging (Anti-Skew Design)

**Core Principle:** Preprocessor and Model are **inseparable, versioned together**.

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":40,"rankSpacing":50,"curve":"linear","padding":8},
  "themeVariables": {
    "primaryColor":"#e3f2fd",
    "lineColor":"#2196f3",
    "edgeLabelBackground":"#fff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"14px",
    "fontWeight":"700"
  }
}}%%

flowchart LR
    %% Stronger Styles
    classDef training fill:#bbdefb,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef artifact fill:#e1bee7,stroke:#2196f3,stroke-width:4px,color:#000,font-weight:800
    classDef component fill:#c8e6c9,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef serving fill:#ffccbc,stroke:#2196f3,stroke-width:3px,color:#000,font-weight:800
    classDef note fill:#ffffff,stroke:#2196f3,stroke-width:2px,color:#000,font-weight:600,font-size:13px,stroke-dasharray:4 4

    %% Training Phase
    subgraph Training["Training Phase"]
        TrainData["Training Data"]:::training
        FitPreproc["Fit Preprocessor<br/>Learn: mean, std, encoders"]:::training
        TrainModel["Train Model<br/>XGBoost/TFT/NLinear"]:::training
    end

    %% Artifact Package
    subgraph ArtifactPackage["Single Artifact (Atomic Unit)"]
        Preprocessor["Stateful Preprocessor<br/>- Fitted transformers<br/>- Imputers<br/>- Scalers<br/>- Encoders"]:::component
        Model["Trained Model<br/>- Weights<br/>- Hyperparams<br/>- Architecture"]:::component
        Wrapper["PyFuncWrapper<br/>Version: 1.2.3<br/>Alias: production"]:::artifact

        Preprocessor ==> Wrapper
        Model ==> Wrapper
    end

    %% Serving Phase
    subgraph Serving["Serving Phase"]
        RawInput["Raw Input<br/>(Same format as training)"]:::serving
        LoadArtifact["Load Artifact<br/>from MLflow"]:::serving
        AutoPreprocess["Auto Apply<br/>Preprocessing"]:::serving
        Inference["Model Inference"]:::serving
        Output["Predictions"]:::serving
    end

    %% Flow
    TrainData ==> FitPreproc
    FitPreproc ==> TrainModel
    FitPreproc -.->|Package| Preprocessor
    TrainModel -.->|Package| Model

    Wrapper ==> |Deploy| LoadArtifact
    RawInput ==> LoadArtifact
    LoadArtifact ==> AutoPreprocess
    AutoPreprocess ==> Inference
    Inference ==> Output

    %% Notes
    Note1["Impossible to deploy<br/>mismatched versions"]:::note
    Note2["Zero logic duplication"]:::note
    Note3["Reproducible results"]:::note

    Note1 -.-> Wrapper
    Note2 -.-> AutoPreprocess
    Note3 -.-> Output

    %% Global link emphasis
    linkStyle default stroke:#2196f3,stroke-width:3px
```

**Benefits:**
- **Atomic Deployment**: Preprocessor + Model deploy together
- **Zero Skew**: Same preprocessing in training/serving
- **Versioning**: Single version for entire pipeline
- **Rollback**: Revert to any previous version instantly

---

## 3. Feast Feature Store Integration

**Key Features:**

| Feature | Training (Offline) | Serving (Online) |
|---------|-------------------|------------------|
| **Storage** | Parquet/BigQuery/Snowflake | Redis/DynamoDB/Cassandra |
| **Latency** | Seconds to minutes | <10ms |
| **Data Volume** | TB+ historical data | KB per entity |
| **Use Case** | Backtesting, CV, Training | Real-time inference |
| **Query Type** | Batch, time-travel | Point lookup |

---

## 4. Design Patterns

### 4.1 Strategy Pattern (Feature Retrieval)

**Benefits:**
- **Open-Closed Principle**: Add new strategies without modifying client
- **Single Responsibility**: Each strategy handles one retrieval type
- **Testability**: Easy to mock strategies


### 4.2 Facade Pattern (Simplified Feature Access)

## 5. Distributed Serving Architecture (Ray Serve)

**Key Features:**
- **Independent Scaling**: Each service scales based on its bottleneck
- **Async Non-blocking**: No service blocks others
- **Fault Isolation**: Service crash doesn't affect others
- **Resource Efficiency**: GPU only for inference


## 6. MLflow Model Registry (Deployment Strategy)

**Benefits:**
- **A/B Testing**: Run staging and production in parallel
- **Instant Rollback**: Change alias, no redeployment
- **Zero Downtime**: Alias switch is atomic
- **Traceability**: Every version logged with metrics

---

## 7. Complete End-to-End Flow

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "nodeSpacing": 30,
    "rankSpacing": 40,
    "curve": "basis",
    "padding": 6
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "14px"
  }
}}%%

flowchart TB
    %% Softer, modern card styles
    classDef data fill:#fffde7,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef feast fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef training fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef registry fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.4px,color:#000,font-weight:700,rx:14,ry:14
    classDef serving fill:#fbe9e7,stroke:#d84315,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef monitor fill:#eceff1,stroke:#37474f,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12

    %% Phase 1: Data Ingestion
    subgraph Phase1["Phase 1: Data Ingestion"]
        Raw["Raw Data<br/>CSV/DB/Stream"]:::data
        Ingest["Feature Engineering<br/>& Ingestion Script"]:::data
        FeastOff["Feast Offline Store<br/>(Parquet)"]:::feast
    end

    %% Phase 2: Materialization
    subgraph Phase2["Phase 2: Materialization"]
        Material["Materialize Script<br/>(Scheduled/On-demand)"]:::feast
        FeastOn["Feast Online Store<br/>(Redis)"]:::feast
    end

    %% Phase 3: Training
    subgraph Phase3["Phase 3: Training"]
        Pipeline["Training Pipeline<br/>- Load from Feast<br/>- CV/Tuning<br/>- Preprocessing fit"]:::training
        Bundle["Bundle Artifacts<br/>Preprocessor + Model"]:::training
        Log["Log to MLflow"]:::registry
    end

    %% Phase 4: Deployment
    subgraph Phase4["Phase 4: Deployment"]
        Alias["Set Alias<br/>production/staging"]:::registry
        Deploy["Deploy Service<br/>FastAPI or Ray Serve"]:::serving
        Load["Load Artifacts<br/>from MLflow"]:::serving
    end

    %% Phase 5: Serving
    subgraph Phase5["Phase 5: Serving"]
        Request["API Request<br/>POST /predict/feast/batch"]:::serving
        FetchFeast["Fetch Features<br/>from Feast Online"]:::feast
        Preprocess["Auto Preprocessing<br/>(Loaded artifacts)"]:::serving
        Inference["Model Inference"]:::serving
        Response["JSON Response"]:::serving
    end

    %% Phase 6: Monitoring
    subgraph Phase6["Phase 6: Monitoring"]
        Metrics["Collect Metrics<br/>Latency, Throughput"]:::monitor
        Dashboard["Ray Dashboard<br/>MLflow UI"]:::monitor
        Alert["Alerting<br/>Anomaly detection"]:::monitor
    end

    %% Flow connections (compact + blue edges)
    Raw --> Ingest
    Ingest --> FeastOff

    FeastOff --> Material
    Material --> FeastOn

    FeastOff -->|"Historical"| Pipeline
    Pipeline --> Bundle
    Bundle --> Log

    Log --> Alias
    Alias --> Deploy
    Deploy --> Load

    Request --> FetchFeast
    FetchFeast -->|"Real-time"| FeastOn
    FetchFeast --> Preprocess
    Preprocess --> Inference
    Inference --> Response

    Response --> Metrics
    Metrics --> Dashboard
    Dashboard --> Alert

    %% Feedback loop unchanged
    Alert -.->|"Retrain trigger"| Pipeline

    %% Edge styling for visibility
    linkStyle default stroke:#2196f3,stroke-width:2px

```

**Timeline:**
1. **Data Ingestion**: Daily/Hourly (Batch)
2. **Materialization**: Hourly/Real-time (Sync)
3. **Training**: Weekly/On-demand (CV + Tuning)
4. **Deployment**: After validation (Alias switch)
5. **Serving**: 24/7 (Real-time)
6. **Monitoring**: Continuous (Metrics collection)

---

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

---
