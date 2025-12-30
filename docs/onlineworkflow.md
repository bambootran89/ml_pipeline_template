# Online Workflow (Serving Pipeline)

## Overview

The serving pipeline delivers real-time predictions with:
- **Distributed services**: Ray Serve for independent scaling
- **Feast integration**: Real-time feature retrieval
- **Zero-skew guarantee**: Uses bundled artifacts from training
- **Multi-entity batch**: Efficient batch predictions
- **Async coordination**: Non-blocking I/O

---

## Complete Serving Architecture

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":85,"rankSpacing":110,"curve":"basis"},
  "themeVariables": {"fontFamily":"Inter, Arial, sans-serif","fontSize":"15px","edgeLabelBackground":"#ffffff"}
}}%%

flowchart TB
    %% Modern rounded card styles
    classDef client fill:#fffde7,stroke:#f9a825,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef ingress fill:#fbe9e7,stroke:#d84315,stroke-width:3px,color:#000,font-weight:700,rx:14,ry:14
    classDef service fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12
    classDef storage fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.2px,color:#000,font-weight:600,rx:12,ry:12
    classDef async fill:#eceff1,stroke:#37474f,stroke-width:2px,stroke-dasharray:5,color:#000,font-style:italic,rx:12,ry:12
    classDef monitor fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000,font-weight:600,rx:12,ry:12

    %% Client Layer
    subgraph C["Client Layer"]
        Client["HTTP Client<br/>curl / requests / SDK"]:::client
    end

    %% API Gateway
    subgraph G["API Gateway<br/>(1 replica)"]
        API["ForecastAPI<br/>FastAPI Ingress<br/>• /predict<br/>• /predict/feast<br/>• /predict/feast/batch<br/>• /health"]:::ingress
    end

    %% Ray Serve Cluster
    subgraph R["Ray Serve Cluster<br/>(Distributed Services)"]

        subgraph F["FeastService<br/>(2 replicas)"]
            F1["Replica 1<br/>CPU 0.5 • I/O bound"]:::service
            F2["Replica 2<br/>CPU 0.5 • I/O bound"]:::service
        end

        subgraph P["PreprocessingService<br/>(2 replicas)"]
            P1["Replica 1<br/>CPU 1–2 • Transform bound"]:::service
            P2["Replica 2<br/>CPU 1–2 • Transform bound"]:::service
        end

        subgraph M["ModelService<br/>(1–4 replicas)"]
            M1["Replica 1<br/>GPU 1 or CPU 4 • Inference bound"]:::service
            M2["Replica 2<br/>GPU 1 or CPU 4"]:::service
            M3["Replica 3"]:::service
            M4["Replica 4"]:::service
        end
    end

    %% External Dependencies
    subgraph E["External Dependencies"]
        Feast["Online Store<br/>Redis / DynamoDB<br/>⚡ <10ms reads"]:::storage
        MLflow["MLflow Registry<br/>Artifacts<br/>production alias"]:::storage
    end

    %% Monitoring
    subgraph O["Monitoring & Observability"]
        Metrics["Prometheus<br/>• Latency p50/p95/p99<br/>• Throughput<br/>• Error rate"]:::monitor
        Dashboard["Ray Dashboard<br/>• Requests<br/>• Replica status<br/>• Resource usage"]:::monitor
    end

    %% Async Coordination
    AsyncPool["Async Event Loop<br/>await asyncio.gather()"]:::async

    %% Request Flow
    Client -->|"1. POST /predict/feast/batch"| API
    API -->|"2. fetch_features.remote()"| F1
    API -.->|"LB"| F2
    F1 & F2 -->|"Query"| Feast

    F1 -->|"3. preprocess.remote()"| P1
    P1 -.->|"LB"| P2
    P1 & P2 -->|"Load preprocessor"| MLflow

    P1 -->|"4. predict.remote()"| M1
    P1 -.->|"LB"| M2
    M1 & M2 & M3 & M4 -->|"Load model"| MLflow

    M1 & M2 & M3 & M4 -->|"5. Predictions"| API
    API -->|"6. JSON Response"| Client

    %% Monitoring Flow
    API -.->|"Export metrics"| Metrics
    Metrics -.-> Dashboard
    F1 & F2 -.-> Dashboard
    P1 & P2 -.-> Dashboard
    M1 & M2 & M3 & M4 -.-> Dashboard

    %% Async Coordination Flow
    API -.->|"Non-blocking"| AsyncPool
    AsyncPool -.-> F1
    AsyncPool -.-> P1
    AsyncPool -.-> M1

    %% Cleaner edge styling
    linkStyle default stroke-width:2px,color:#222

```

---

## Request Flow Details

### 1. Traditional Prediction (Data in Payload)

```mermaid
%%{init: {
  "theme": "base",
  "sequence": {
    "mirrorActors": false,
    "actorFontSize": "15px",
    "noteFontSize": "14px",
    "messageFontSize": "15px"
  },
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "secondaryColor": "#FFF8E1",
    "tertiaryColor": "#E8F5E9",
    "noteBkgColor": "#FAFAFA",
    "noteBorderColor": "#90A4AE",
    "activationBorderColor": "#1976D2",
    "activationBkgColor": "#FFFFFF",
    "participantBkgColor": "#FFFFFF",
    "participantBorderColor": "#1976D2",
    "participantFontWeight": "600",
    "fontFamily": "Inter, Arial, sans-serif"
  }
}}%%

sequenceDiagram
    autonumber
    box rgba(227,242,253,0.35) Client Request
    participant Client
    participant API as ForecastAPI
    end

    box rgba(232,245,233,0.35) Data Processing
    participant Preprocess as PreprocessingService
    participant Model as ModelService
    end

    box rgba(243,229,245,0.4) Model Registry
    participant MLflow
    end

    Client->>+API: POST /predict {data}
    Note over API: Schema validation<br/>Request integrity check

    API->>+Preprocess: preprocess.remote(data)
    Note over Preprocess: Load preprocessor<br/>(cached in memory)

    alt Cache miss
        Preprocess->>MLflow: Fetch preprocessor artifact
        MLflow-->>Preprocess: preprocessor.pkl
    else Cache hit
        Note right of Preprocess: Use cached preprocessor
    end

    Preprocess-->>-API: transformed_data

    API->>+Model: predict.remote(transformed_data)
    Note over Model: Load model artifact<br/>(cached in memory)

    alt Cache miss
        Model->>MLflow: Fetch model artifact
        MLflow-->>Model: model.pkl
    else Cache hit
        Note right of Model: Use cached model
    end

    Model-->>-API: predictions
    Note over Model: Run inference<br/>Model execution

    API-->>-Client: JSON response<br/>{prediction: [...]}
    Note over API: Response formatting<br/>Serializable output

```

**Latency Breakdown:**
- Validation: ~1ms
- Preprocessing: ~50ms
- Model inference: ~30ms
- **Total: ~80ms** (after cache warmup)

---

### 2. Feast-Native Prediction (Single Entity)

```mermaid
%%{init: {
  "theme": "base",
  "sequence": {
    "actorFontSize": "15px",
    "messageFontSize": "15px",
    "noteFontSize": "14px",
    "mirrorActors": false
  },
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "secondaryColor": "#C8E6C9",
    "tertiaryColor": "#FFF8E1",
    "noteBkgColor": "#FAFAFA",
    "noteBorderColor": "#90A4AE",
    "activationBorderColor": "#1976D2",
    "activationBkgColor": "#FFFFFF",
    "participantBkgColor": "#FFFFFF",
    "participantBorderColor": "#1976D2",
    "participantFontWeight": "600",
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "14px"
  }
}}%%

sequenceDiagram
    autonumber

    %% Layer grouping for clarity
    box rgba(255,248,225,0.35) Client Layer
      participant Client
      participant API as ForecastAPI
    end

    box rgba(200,230,201,0.35) Feature Store Layer
      participant Feast as FeastService
      participant FeastStore as Feast Online Store
    end

    box rgba(227,242,253,0.3) Model Processing
      participant Preprocess as PreprocessingService
      participant Model as ModelService
    end

    %% Request flow
    Client->>+API: POST /predict/feast {time_point, entities}

    Note over API: Request validation<br/>Schema & payload integrity

    API->>+Feast: fetch_features.remote(time_point, entities)
    Note over Feast: Feature retrieval orchestration<br/>Uses in-memory cache when available

    Feast->>+FeastStore: get_online_features(entity_rows)
    Note over FeastStore: Ultra-fast lookup<br/>Key-value read (sub-10ms)

    FeastStore-->>-Feast: feature_dict

    Feast-->>-API: features_df
    Note right of API: Returns structured<br/>DataFrame response

    API->>+Preprocess: preprocess.remote(features_df)
    Note over Preprocess: Stateful transformations<br/>Impute → Scale → Encode

    Preprocess-->>-API: transformed_df

    API->>+Model: predict.remote(transformed_df)
    Note over Model: Model execution<br/>CPU/GPU inference

    Model-->>-API: predictions

    API-->>-Client: JSON response {prediction: [...]}
    Note over API: Response serialization<br/>Ensures deterministic output format

```

**Latency Breakdown:**
- Validation: ~1ms
- Feast query: ~10ms (Redis)
- Preprocessing: ~50ms
- Model inference: ~30ms
- **Total: ~90ms**

**Key Advantage:** Client doesn't need to fetch features!

---

### 3. Batch Prediction (Multi-Entity)

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#E3F2FD",
    "secondaryColor": "#C8E6C9",
    "tertiaryColor": "#FFF8E1",
    "noteBkgColor": "#FAFAFA",
    "noteBorderColor": "#90A4AE",
    "participantBkgColor": "#FFFFFF",
    "participantBorderColor": "#1976D2",
    "fontFamily": "Inter, Arial, sans-serif",
    "fontSize": "14px"
  }
}}%%

sequenceDiagram
    autonumber

    participant Client
    participant API as ForecastAPI
    participant Feast as FeastService
    participant FeastStore as FeastOnlineStore
    participant Preprocess as PreprocessingService
    participant Model as ModelService

    Client->>+API: POST /predict/feast/batch {time_point="now", entities=[1,2,3,4,5]}
    Note over API: Validate request schema

    API->>+Feast: fetch_features.remote(time_point="now", entities=[1..5])
    Feast->>+FeastStore: get_online_features(entity_rows)
    Note over FeastStore: Fast key-value lookup
    FeastStore-->>-Feast: features_dict_all
    Feast-->>-API: features_df_all

    par Parallel processing
        API->>+Preprocess: preprocess.remote(entity_1_df)
        Preprocess-->>-API: transformed_1
        API->>+Model: predict.remote(transformed_1)
        Model-->>-API: pred_1
    and
        API->>+Preprocess: preprocess.remote(entity_2_df)
        Preprocess-->>-API: transformed_2
        API->>+Model: predict.remote(transformed_2)
        Model-->>-API: pred_2
    and
        Note over API: Entity 3-5 follow same flow
    end

    Note over API: Collect all predictions (async gather)
    API-->>-Client: {"predictions": {"1":[...],"2":[...],"..."}}
    Note over Client: Response received

```

**Performance:**
- **Single Feast query** for all entities (efficient!)
- **Parallel preprocessing** and inference
- **5 entities processed in ~200ms** (vs 450ms sequential)

**Scaling:** 100 entities in ~500ms (Ray auto-scales services)

---

## Service Responsibilities

```mermaid
%%{init: {
  "theme": "default",
  "flowchart": {
    "nodeSpacing": 70,
    "rankSpacing": 90,
    "curve": "linear"
  },
  "themeVariables": {
    "lineColor": "#222222",
    "edgeLabelBackground": "#ffffff",
    "fontSize": "16px",
    "primaryBorderColor": "#222222",
    "primaryTextColor": "#000000"
  }
}}%%

flowchart TB
    classDef api fill:#ffccbc,stroke:#d84315,stroke-width:2px,color:#000
    classDef feast fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef preprocess fill:#bbdefb,stroke:#1565c0,stroke-width:2px,color:#000
    classDef model fill:#e1bee7,stroke:#6a1b9a,stroke-width:2px,color:#000

    subgraph API["ForecastAPI (Ingress)"]
        APIResp["HTTP routing<br/>Request validation<br/>Response formatting<br/>Error handling<br/>Async coordination"]:::api
    end

    subgraph Feast["FeastService"]
        FeastResp["Feature retrieval<br/>Entity row build<br/>DataFrame convert<br/>Error handling<br/><br/>I/O bound • 2 replicas • 0.5 CPU"]:::feast
    end

    subgraph Preprocess["PreprocessingService"]
        PrepResp["Load from MLflow<br/>Transform<br/>Missing values<br/>Scaling<br/><br/>CPU bound • 2 replicas • 1-2 CPU"]:::preprocess
    end

    subgraph Model["ModelService"]
        ModelResp["Load model<br/>Build tensors<br/>Inference<br/>Return preds<br/><br/>GPU/CPU bound • 1-4 replicas"]:::model
    end

    API --> Feast
    API --> Preprocess
    API --> Model

```

---

## Auto-Scaling Configuration

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":65,"rankSpacing":85,"curve":"basis"},
  "themeVariables": {
    "primaryColor":"#e3f2fd",
    "edgeLabelBackground":"#ffffff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"15px",
    "lineColor":"#000000",
    "textColor":"#000000"
  }
}}%%

flowchart TB
    %% Styles (modern cards)
    classDef metric fill:#fff59d,stroke:#f9a825,stroke-width:2.2px,color:#000,font-weight:600,rx:10,ry:10
    classDef decision fill:#ffab91,stroke:#b71c1c,stroke-width:3px,color:#000,font-weight:700,rx:12,ry:12
    classDef action fill:#a5d6a7,stroke:#1b5e20,stroke-width:2.2px,color:#000,font-weight:600,rx:10,ry:10

    %% Metrics
    subgraph Metrics["Collected Metrics"]
        style Metrics stroke:#000000,stroke-width:2.5px,fill:#fafafa,color:#000,font-weight:700,font-size:16px
        CPU["CPU Usage<br/>per service"]:::metric
        Memory["Memory Usage<br/>per service"]:::metric
        QueueDepth["Request Queue Depth<br/>per service"]:::metric
        Latency["Response Latency<br/>p95, p99"]:::metric
    end

    %% Decision Engine
    subgraph Decision["Ray Autoscaler Decision Engine"]
        style Decision stroke:#b71c1c,stroke-width:3px,fill:#ffebee,color:#000,font-weight:700,font-size:16px
        Rules["Scaling Rules:<br/>CPU > 70% → Scale up<br/>Queue > 10 → Scale up<br/>CPU < 30% 5min → Scale down<br/>Min: 1 · Max: 10 replicas"]:::decision
    end

    %% Actions
    subgraph Actions["Scaling Actions"]
        style Actions stroke:#000000,stroke-width:2.5px,fill:#fafafa,color:#000,font-weight:700,font-size:16px
        ScaleUp["Scale Up<br/>Add replica"]:::action
        ScaleDown["Scale Down<br/>Remove replica"]:::action
        NoOp["No Action<br/>Capacity OK"]:::action
    end

    %% Flow
    CPU --> Rules
    Memory --> Rules
    QueueDepth --> Rules
    Latency --> Rules

    Rules --> ScaleUp
    Rules --> ScaleDown
    Rules --> NoOp

    %% Link visibility improved
    linkStyle default stroke:#000000,stroke-width:2.2px
```


---

## Artifact Loading & Caching

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":60,"rankSpacing":90,"curve":"basis"},
  "themeVariables": {
    "lineColor":"#000000",
    "edgeLabelBackground":"#ffffff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"14px",
    "textColor":"#000000"
  }
}}%%

flowchart TB

    %% Styles
    classDef cold fill:#bbdefb,stroke:#1976d2,stroke-width:2.4px,color:#000,font-weight:600,rx:10,ry:10
    classDef warm fill:#c8e6c9,stroke:#388e3c,stroke-width:2.4px,color:#000,font-weight:600,rx:10,ry:10
    classDef hot fill:#ffccbc,stroke:#e64a19,stroke-width:2.4px,color:#000,font-weight:600,rx:10,ry:10
    classDef note fill:#fafafa,stroke:#424242,stroke-width:1.8px,color:#000,font-weight:500,rx:8,ry:8,stroke-dasharray:4 2

    %% Cold Start block
    Request1["Request 1<br/>"]:::cold
    LoadMLflow1["Load from MLflow<br/>production alias"]:::cold
    Download1["Download artifacts<br/>preprocessor.pkl + model.pkl"]:::cold
    Cache1["Cache in memory"]:::cold
    Predict1["Run prediction<br/>~500ms total"]:::cold
    Note1["First request slow<br/>Artifact download"]:::note

    %% Warm Start block
    Request2["Request 2+"]:::warm
    CheckCache["Check memory cache"]:::warm
    UseCache["Use cached artifacts"]:::warm
    Predict2["Run prediction<br/>~80ms total"]:::warm
    Note2["Cached requests fast<br/>No download"]:::note

    %% Hot Path block
    RequestN["Request N"]:::hot
    DirectInf["Direct inference<br/>No loading"]:::hot
    PredictN["Run prediction<br/>~50ms"]:::hot
    Note3["Pre-warmed optimal<br/>Model in GPU memory"]:::note

    %% Flow xuống dọc
    Request1 --> LoadMLflow1 --> Download1 --> Cache1 --> Predict1
    Predict1 --> Request2 --> CheckCache --> UseCache --> Predict2
    Predict2 --> RequestN --> DirectInf --> PredictN

    %% Notes đứng riêng không nối flow để tránh lỗi render
    ColdNote1[" "]
    WarmNote1[" "]
    HotNote1[" "]

    %% Connect notes visually bằng comment grouping
    Request1 --- Note1
    Request2 --- Note2
    RequestN --- Note3

```

**Pre-warming Strategy:**
```python
# In service __init__
async def _warmup(self):
    """Load artifacts during service startup"""
    if self.mlflow_manager.enabled:
        model_name = self.cfg.experiment["model"].lower()

        # Load preprocessor
        self.preprocessor = self.mlflow_manager.load_component(
            name=f"{model_name}_preprocessor",
            alias="production"  # Always load production
        )

        # Load model
        self.model = self.mlflow_manager.load_component(
            name=f"{model_name}_model",
            alias="production"
        )

        self.ready = True
        print("[Service] Pre-warmed and ready")
```

---

## Monitoring & Observability

### Ray Dashboard (Built-in)

```
http://localhost:8265

Dashboard Metrics:
├── Cluster Overview
│   ├── Total nodes: 3
│   ├── Total CPUs: 24
│   └── Total GPUs: 2
│
├── Deployments
│   ├── ForecastAPI
│   │   ├── Replicas: 1/1
│   │   └── Requests: 1,234 total, 45 RPS
│   │
│   ├── FeastService
│   │   ├── Replicas: 2/2
│   │   ├── CPU usage: 45%
│   │   └── Queue depth: 0
│   │
│   ├── PreprocessingService
│   │   ├── Replicas: 2/2
│   │   ├── CPU usage: 68%
│   │   └── Queue depth: 2
│   │
│   └── ModelService
│       ├── Replicas: 1/1
│       ├── GPU usage: 82%
│       └── Latency p95: 45ms
│
└── Metrics
    ├── Request latency
    │   ├── p50: 80ms
    │   ├── p95: 120ms
    │   └── p99: 180ms
    │
    └── Throughput
        └── 45 requests/sec
```

### Custom Metrics

```python
# Export to Prometheus
from prometheus_client import Counter, Histogram

# Request counter
requests_total = Counter(
    'forecast_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

# Latency histogram
request_duration = Histogram(
    'forecast_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

# In endpoint
@app.post("/predict/feast")
async def predict_feast(request):
    with request_duration.labels(endpoint="feast").time():
        try:
            result = await process(request)
            requests_total.labels(
                endpoint="feast",
                status="success"
            ).inc()
            return result
        except Exception as e:
            requests_total.labels(
                endpoint="feast",
                status="error"
            ).inc()
            raise
```

---

## Deployment Comparison

### FastAPI (Simple)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":60,"rankSpacing":80,"curve":"basis"},
  "themeVariables": {
    "primaryColor":"#e3f2fd",
    "primaryBorderColor":"#1a237e",
    "lineColor":"#1a237e",
    "edgeLabelBackground":"#ffffff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"15px",
    "fontWeight":"600",
    "textColor":"#000000"
  }
}}%%

flowchart TB
    %% Styles
    classDef simple fill:#c8e6c9,stroke:#1b5e20,stroke-width:2.5px,color:#000,font-weight:600,rx:10,ry:10

    %% FastAPI
    subgraph FastAPI["FastAPI (Single Process)"]
        style FastAPI stroke:#1a237e,stroke-width:3px,fill:#f5faff,font-size:16px,font-weight:700,color:#000
        App["FastAPI App<br/>- All logic in-process<br/>- Synchronous<br/>- Simple deployment"]:::simple

        subgraph Logic["In-Process Components"]
            style Logic stroke:#1a237e,stroke-width:2.5px,fill:#ffffff,font-size:15px,font-weight:700,color:#000
            Service["ModelsService<br/>- Model<br/>- Preprocessor<br/>- Feast facade"]:::simple
        end
    end

    Client["HTTP Client"]:::simple
    Client --> App
    App --> Service

    Metrics["Metrics:<br/>- Memory: ~500MB<br/>- Throughput: ~10 RPS<br/>- Latency: ~100ms<br/>- Scaling: Manual (gunicorn)"]:::simple

    Service -.-> Metrics

    %% Link style
    linkStyle default stroke:#000000,stroke-width:2.3px,color:#000

```

**Use Cases:**
- Development/Testing
- Low traffic (<100 RPS)
- Simple deployment
- No auto-scaling
- No GPU optimization

---

### Ray Serve (Production)

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {"nodeSpacing":60,"rankSpacing":80,"curve":"basis"},
  "themeVariables": {
    "primaryColor":"#e3f2fd",
    "primaryBorderColor":"#0d47a1",
    "lineColor":"#0d47a1",
    "edgeLabelBackground":"#fff",
    "fontFamily":"Inter, Arial, sans-serif",
    "fontSize":"15px",
    "fontWeight":"600",
    "textColor":"#000"
  }
}}%%

flowchart TB
    %% Styles
    classDef advanced fill:#bbdefb,stroke:#0d47a1,stroke-width:2.6px,color:#000,font-weight:600,rx:10,ry:10

    %% Ray Serve
    Client["HTTP Client"]:::advanced

    subgraph RayServe["Ray Serve (Distributed Cluster)"]
        style RayServe stroke:#0d47a1,stroke-width:3.2px,fill:#f5faff,font-size:16px,font-weight:700,color:#000
        Ingress["ForecastAPI<br/>HTTP Ingress"]:::advanced

        subgraph Services["Distributed Services"]
            style Services stroke:#1565c0,stroke-width:2.8px,fill:#fff,font-size:15px,font-weight:700,color:#000
            Feast["FeastService<br/>2 replicas<br/>Auto-scale"]:::advanced
            Preprocess["PreprocessingService<br/>2 replicas<br/>Auto-scale"]:::advanced
            Model["ModelService<br/>1-4 replicas<br/>GPU support"]:::advanced
        end
    end

    Client --> Ingress
    Ingress --> Feast
    Ingress --> Preprocess
    Ingress --> Model

    Metrics["Metrics:<br/>- Memory: ~2GB total<br/>- Throughput: ~500 RPS<br/>- Latency: ~80ms<br/>- Scaling: Automatic<br/>- GPU: Efficient"]:::advanced

    Services -.-> Metrics

    %% Link style
    linkStyle default stroke:#000,stroke-width:2.2px,color:#000

```

**Use Cases:**
- Production serving
- High traffic (>100 RPS)
- Auto-scaling needed
- GPU inference
- Multi-model serving
- Higher complexity

---

## Summary

**Key Features:**
- **Distributed**: Ray Serve for independent scaling
- **Fast**: Async coordination, sub-100ms latency
- **Zero-skew**: Bundled artifacts from training
- **Multi-entity**: Efficient batch predictions
- **Observable**: Built-in metrics and dashboards

**Deployment Command:**
```bash
# Ray Serve
python mlproject/serve/ray_deploy.py

# OR FastAPI (simpler)
python mlproject/serve/api.py
```

**Endpoints:**
- `POST /predict` - Traditional (data in payload)
- `POST /predict/feast` - Feast single/multi-entity
- `POST /predict/feast/batch` - Batch multi-entity (efficient)
- `GET /health` - Health check

**Serving Flow:**
1. Load artifacts from MLflow (cached)
2. Fetch features from Feast Online Store
3. Apply preprocessing (bundled with model)
4. Run inference
5. Return predictions

**Performance:**
- Single prediction: ~80ms
- Batch (5 entities): ~200ms
- Batch (100 entities): ~500ms
- Throughput: 50-500 RPS (Ray Serve)
