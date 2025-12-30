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
  "flowchart": {"nodeSpacing":95,"rankSpacing":120,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","lineColor":"#000","edgeStrokeWidth":"3px","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"600","clusterBorderRadius":"10px","nodeBorderRadius":"8px"}
}}%%

flowchart TB
    %% Styles
    classDef dataSource fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef featureStore fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef training fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef registry fill:#e1bee7,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef serving fill:#ffccbc,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef monitoring fill:#b0bec5,stroke:#000,stroke-width:3px,color:#000,font-weight:700

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

    %% Connections (bold main flow)
    RawData ==> Feast:::dataSource
    Feast ==> Pipeline:::featureStore
    Pipeline ==> Artifacts:::training
    Artifacts ==> MLflow:::training

    MLflow ==> FastAPI:::registry
    MLflow ==> RayServe:::registry

    Feast ==> FastAPI:::featureStore
    Feast ==> RayServe:::featureStore

    FastAPI ==> Metrics:::serving
    RayServe ==> Metrics:::serving
    Metrics ==> Dashboard:::monitoring

    %% Edge emphasis
    linkStyle default stroke:#000,stroke-width:3px
```

---

## 2. Unified Artifact Packaging (Anti-Skew Design)

**Core Principle:** Preprocessor and Model are **inseparable, versioned together**.

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":90,"rankSpacing":110,"curve":"linear","padding":12},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","lineColor":"#000","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"700"}
}}%%

flowchart LR
    %% Stronger Styles
    classDef training fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef artifact fill:#e1bee7,stroke:#000,stroke-width:4px,color:#000,font-weight:800
    classDef component fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef serving fill:#ffccbc,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef note fill:#ffffff,stroke:#000,stroke-width:2px,color:#000,font-weight:600,font-size:13px,stroke-dasharray:4 4

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
    linkStyle default stroke:#000,stroke-width:3px
```

**Benefits:**
- **Atomic Deployment**: Preprocessor + Model deploy together
- **Zero Skew**: Same preprocessing in training/serving
- **Versioning**: Single version for entire pipeline
- **Rollback**: Revert to any previous version instantly

---

## 3. Feast Feature Store Integration

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":110,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"800","lineColor":"#000"}
}}%%

flowchart TB
    %% Stronger Styles
    classDef source fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef feast fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef client fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef note fill:#ffffff,stroke:#000,stroke-width:2px,color:#000,font-weight:600,stroke-dasharray:5 5,font-size:13px

    %% Data Sources
    subgraph Sources["Data Sources"]
        DB["PostgreSQL<br/>MySQL"]:::source
        Files["Parquet<br/>CSV"]:::source
        Stream["Kafka<br/>Kinesis"]:::source
    end

    %% Feast Core
    subgraph FeastCore["Feast Feature Store"]
        Registry["Feature Registry<br/>- Definitions<br/>- Metadata<br/>- Schemas"]:::feast

        subgraph Stores["Storage Backends"]
            Offline["Offline Store<br/>(Parquet/BigQuery)<br/>Training & Backfill"]:::feast
            Online["Online Store<br/>(Redis/DynamoDB)<br/>Low-latency serving<br/>&lt;10ms reads"]:::feast
        end

        Materialize["Materialization<br/>Offline → Online<br/>(Scheduled/On-demand)"]:::feast
    end

    %% Clients
    subgraph Clients["Client Applications"]
        Training["Training Pipeline<br/>Historical features<br/>Point-in-time join"]:::client
        Serving["Serving API<br/>Real-time features<br/>Multi-entity batch"]:::client
    end

    %% Flows (bold for primary paths)
    DB ==> Registry
    Files ==> Registry
    Stream ==> Registry

    Registry ==> Offline
    Offline ==> |Materialize| Online
    Materialize -.-> |Schedule| Online

    Offline ==> |get_historical_features| Training
    Online ==> |get_online_features| Serving

    %% Notes / Annotations
    Note1["Time-travel:<br/>Point-in-time correctness"]:::note
    Note2["Sub-10ms latency:<br/>Production serving"]:::note
    Note3["Consistency:<br/>Same features training/serving"]:::note

    Note1 -.-> Offline
    Note2 -.-> Online
    Note3 -.-> Materialize

    %% Global edge emphasis
    linkStyle default stroke:#000,stroke-width:3px
```

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

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":110,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"800","lineColor":"#000"}
}}%%

flowchart TB
    %% Stronger Styles (higher contrast + bold text)
    classDef client fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef factory fill:#e1bee7,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef interface fill:#ffccbc,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef strategy fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef decision fill:#ffffff,stroke:#000,stroke-width:2px,color:#000,font-weight:600,font-size:14px

    %% Client
    Client["Client Code<br/>(Facade/Pipeline)"]:::client

    %% Factory
    Factory["StrategyFactory<br/>.create(data_type, mode, time_point)"]:::factory

    %% Interface
    Interface["FeatureRetrievalStrategy<br/>«interface»<br/>+ retrieve()"]:::interface

    %% Concrete Strategies
    Timeseries["TimeseriesRetrievalStrategy<br/>- get_sequence_by_range()<br/>- Windowed data"]:::strategy
    Tabular["TabularRetrievalStrategy<br/>- get_historical_features()<br/>- Entity joins"]:::strategy
    Online["OnlineRetrievalStrategy<br/>- get_online_features()<br/>- get_latest_n_sequence()<br/>- Multi-entity support"]:::strategy

    %% Decision Logic Node (fixed syntax + clean formatting)
    Decision{"mode == 'online'?<br/>data_type?"}:::decision

    %% Relationships (primary flow emphasized with ==>)
    Client ==>|"1. Request strategy"| Factory
    Factory ==>|"2. Create"| Interface

    %% Interface implementation links (fixed direction and syntax)
    Interface -.-> Timeseries
    Interface -.-> Tabular
    Interface -.-> Online

    %% Return paths
    Timeseries ==>|"3. Return"| Factory
    Tabular ==>|"3. Return"| Factory
    Online ==>|"3. Return"| Factory

    %% Factory usage
    Factory ==>|"4. Use"| Client

    %% Decision branching (connect directly to strategies)
    Factory -.-> Decision
    Decision -.->|"online"| Online
    Decision -.->|"historical + timeseries"| Timeseries
    Decision -.->|"historical + tabular"| Tabular

    %% Global edge emphasis for better visibility
    linkStyle default stroke:#000,stroke-width:3px
```

**Code Example:**
```python
# Client code - NO conditionals!
strategy = StrategyFactory.create(
    data_type="timeseries",
    mode="online",
    time_point="now"
)

df = strategy.retrieve(
    store=feast_store,
    features=["temp", "humidity"],
    entity_key="location_id",
    entity_id=42,
    config=config
)
```

**Benefits:**
- **Open-Closed Principle**: Add new strategies without modifying client
- **Single Responsibility**: Each strategy handles one retrieval type
- **Testability**: Easy to mock strategies

---

### 4.2 Facade Pattern (Simplified Feature Access)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":90,"rankSpacing":120,"curve":"linear","padding":12},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"700","lineColor":"#000"}
}}%%

flowchart LR
    %% Stronger contrast styles
    classDef client fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef facade fill:#e1bee7,stroke:#000,stroke-width:4px,color:#000,font-weight:900
    classDef subsystem fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef edgeMain stroke:#000,stroke-width:3.2px,font-weight:800,color:#000

    %% Simple Interface
    subgraph Simple["Simple Interface (3 lines)"]
        Client["Client Code"]:::client
        Facade["FeatureStoreFacade<br/><br/>load_features(<br/>  time_point,<br/>  entities<br/>)"]:::facade
    end

    %% Complex Subsystem
    subgraph Complex["Complex Subsystem (Hidden)"]
        Factory["FeatureStoreFactory<br/>Initialize store"]:::subsystem
        Strategy["StrategyFactory<br/>Select strategy"]:::subsystem
        Store["Feast FeatureStore<br/>Complex API"]:::subsystem
        Config["Config Resolution<br/>Entities, features"]:::subsystem
        Error["Error Handling<br/>Retries, fallbacks"]:::subsystem
    end

    %% Primary flow emphasized
    Client ==>|"Simple call"| Facade:::edgeMain
    Facade ==>|"1. Create store"| Factory:::edgeMain
    Facade ==>|"2. Select strategy"| Strategy:::edgeMain
    Facade ==>|"3. Load features"| Store:::edgeMain
    Facade ==>|"4. Resolve config"| Config:::edgeMain
    Facade ==>|"5. Handle errors"| Error:::edgeMain

    %% Encapsulation return paths
    Factory -.->|"Encapsulated"| Facade
    Strategy -.->|"Encapsulated"| Facade
    Store -.->|"Encapsulated"| Facade
    Config -.->|"Encapsulated"| Facade
    Error -.->|"Encapsulated"| Facade

    %% Final output flow preserved
    Facade ==>|"DataFrame"| Client:::edgeMain

    %% Global edge visibility boost
    linkStyle default stroke:#000,stroke-width:2.8px
```

**Before (Complex):**
```python
# 30 lines of boilerplate
from feast import FeatureStore
store = FeatureStore(repo_path="...")
features = [f"{view}:{f}" for f in feature_list]
entity_rows = [{entity_key: eid} for eid in entity_ids]

if mode == "online":
    result = store.get_online_features(
        entity_rows=entity_rows,
        features=features
    )
else:
    entity_df = pd.DataFrame({...})
    result = store.get_historical_features(
        entity_df=entity_df,
        features=features
    )
# ... error handling, parsing, etc.
```

**After (Simple):**
```python
# 3 lines!
facade = FeatureStoreFacade(config, mode="online")
df = facade.load_features(time_point="now", entities=[1, 2, 3])
```

---

## 5. Distributed Serving Architecture (Ray Serve)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":100,"rankSpacing":130,"curve":"linear","padding":14},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#fff","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"700","lineColor":"#000"}
}}%%

flowchart TB
    %% Stronger contrast & sharper lines
    classDef client fill:#fff9c4,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef ingress fill:#ffccbc,stroke:#000,stroke-width:4px,color:#000,font-weight:900
    classDef service fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef storage fill:#e1bee7,stroke:#000,stroke-width:3px,color:#000,font-weight:700
    classDef async fill:#b0bec5,stroke:#000,stroke-width:2.5px,stroke-dasharray:5,color:#000,font-weight:700
    classDef edgeMain stroke:#000,stroke-width:3.2px,font-weight:800,color:#000

    %% Client
    Client["HTTP Client<br/>POST /predict/feast/batch"]:::client

    %% Ingress
    subgraph Ingress["API Gateway (1 replica)"]
        API["ForecastAPI<br/>- Request validation<br/>- Response formatting<br/>- Error handling"]:::ingress
    end

    %% Services Layer
    subgraph Services["Distributed Services (Ray Cluster)"]
        subgraph FeastSvc["FeastService (2 replicas)"]
            F1["Replica 1<br/>I/O bound"]:::service
            F2["Replica 2<br/>I/O bound"]:::service
        end

        subgraph PreprocessSvc["PreprocessingService (2 replicas)"]
            P1["Replica 1<br/>CPU bound"]:::service
            P2["Replica 2<br/>CPU bound"]:::service
        end

        subgraph ModelSvc["ModelService (1-4 replicas)"]
            M1["Replica 1<br/>GPU/CPU"]:::service
            M2["Replica 2<br/>GPU/CPU"]:::service
        end
    end

    %% Storage
    subgraph Storage["External Dependencies"]
        Feast["Feast<br/>Feature Store"]:::storage
        MLflow["MLflow<br/>Model Registry"]:::storage
    end

    %% Primary flow (kept identical but highlighted visually)
    Client ==>|"1. HTTP Request"| API:::edgeMain

    API ==>|"2. async fetch_features.remote()"| F1:::edgeMain
    API -.->|"Load balance"| F2
    F1 ==>|"Query"| Feast:::edgeMain
    F2 -->|"Query"| Feast

    F1 ==>|"3. async preprocess.remote()"| P1:::edgeMain
    F1 -.->|"Load balance"| P2
    P1 ==>|"Load artifacts"| MLflow:::edgeMain

    P1 ==>|"4. async predict.remote()"| M1:::edgeMain
    P1 -.->|"Load balance"| M2
    M1 ==>|"Load model"| MLflow:::edgeMain

    M1 ==>|"5. Predictions"| API:::edgeMain
    API ==>|"6. JSON Response"| Client:::edgeMain

    %% Async coordination
    AsyncCoord["Async Coordination<br/>(await asyncio.gather)"]:::async
    API -.->|"Non-blocking"| AsyncCoord
    AsyncCoord -.-> F1
    AsyncCoord -.-> P1
    AsyncCoord -.-> M1

    %% Edge visibility boost globally
    linkStyle default stroke:#000,stroke-width:2.8px

```

**Service Scaling Matrix:**

| Service | Bottleneck | Replicas | Resource | Auto-scale Trigger |
|---------|-----------|----------|----------|-------------------|
| **ForecastAPI** | HTTP routing | 1 | 0.5 CPU | Request queue |
| **FeastService** | I/O latency | 2-4 | 0.5 CPU | Feast query time |
| **PreprocessingService** | CPU computation | 2-4 | 1-2 CPU | CPU usage >70% |
| **ModelService** | GPU/CPU inference | 1-4 | 1 GPU or 4 CPU | GPU usage >80% |

**Key Features:**
- **Independent Scaling**: Each service scales based on its bottleneck
- **Async Non-blocking**: No service blocks others
- **Fault Isolation**: Service crash doesn't affect others
- **Resource Efficiency**: GPU only for inference

---

## 6. MLflow Model Registry (Deployment Strategy)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":90,"rankSpacing":110,"curve":"linear","padding":12},
  "themeVariables": {"primaryColor":"#e3f2fd","edgeLabelBackground":"#ffffff","fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","fontWeight":"700","lineColor":"#000000"}
}}%%

flowchart LR
    %% Stronger styles for visibility
    classDef training fill:#c8e6c9,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef registry fill:#e1bee7,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef alias fill:#ffccbc,stroke:#000,stroke-width:4px,color:#000,font-weight:900
    classDef serving fill:#bbdefb,stroke:#000,stroke-width:3px,color:#000,font-weight:800
    classDef edgeMain stroke:#000,stroke-width:3.4px,font-weight:800,color:#000

    %% Training
    subgraph Train["Training Pipeline"]
        direction LR
        T1["Train Model v1"]:::training
        T2["Train Model v2"]:::training
        T3["Train Model v3"]:::training
    end

    %% Registry
    subgraph Registry["MLflow Model Registry\nModel: 'xgboost_forecaster'"]
        direction LR
        V1["Version 1\nAccuracy: 92%\nDate: 2024-01-01"]:::registry
        V2["Version 2\nAccuracy: 94%\nDate: 2024-01-10"]:::registry
        V3["Version 3\nAccuracy: 95%\nDate: 2024-01-15"]:::registry
    end

    %% Aliases
    subgraph Aliases["Deployment Aliases"]
        direction LR
        Latest["latest\n(newest)"]:::alias
        Staging["staging\n(testing)"]:::alias
        Production["production\n(live traffic)"]:::alias
    end

    %% Serving
    subgraph Serve["Serving Environments"]
        direction LR
        Dev["Dev API\n--alias latest"]:::serving
        Test["Test API\n--alias staging"]:::serving
        Prod["Prod API\n--alias production"]:::serving
    end

    %% Main flow with stronger arrows
    T1 ==>|"Log"| V1:::edgeMain
    T2 ==>|"Log"| V2:::edgeMain
    T3 ==>|"Log"| V3:::edgeMain

    V3 -.->|"Auto-assign"| Latest
    V2 -.->|"Promote"| Staging
    V1 -.->|"Promote"| Production

    Latest ==>|"Load"| Dev:::edgeMain
    Staging ==>|"Load"| Test:::edgeMain
    Production ==>|"Load"| Prod:::edgeMain

    %% Promotion workflow
    Promote["Promotion Workflow\nstaging → production"]:::alias
    V2 -.->|"After validation"| Promote
    Promote -.->|"mlflow.set_alias()"| Production

    %% Global edge visibility boost
    linkStyle default stroke:#000,stroke-width:2.9px,color:#000,font-weight:700

```

**Deployment Workflow:**

```python
# 1. Train new model → Auto-assigned to "latest"
mlflow.log_model(..., registered_model_name="xgboost_forecaster")
# Version 3 created, alias="latest" assigned

# 2. Test in staging
mlflow.set_registered_model_alias(
    name="xgboost_forecaster",
    alias="staging",
    version=3
)
# Load in test environment
model = mlflow.pyfunc.load_model("models:/xgboost_forecaster@staging")

# 3. Promote to production (after validation)
mlflow.set_registered_model_alias(
    name="xgboost_forecaster",
    alias="production",
    version=3
)
# Live traffic now uses v3

# 4. Rollback if needed (instant!)
mlflow.set_registered_model_alias(
    name="xgboost_forecaster",
    alias="production",
    version=2  # Revert to previous version
)
```

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
    "nodeSpacing": 70,
    "rankSpacing": 90,
    "curve": "basis"
  },
  "themeVariables": {
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "15px"
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

    %% Flow connections (same logic, cleaner arrows)
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

    %% Edge styling for better visibility
    linkStyle default stroke-width:2px,color:#000

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
