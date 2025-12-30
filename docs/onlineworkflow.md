# Online Workflow (Serving Pipeline)

## Overview

The serving pipeline delivers real-time predictions with:
- **Distributed services**: Ray Serve for independent scaling
- **Feast integration**: Real-time feature retrieval
- **Zero-skew guarantee**: Uses bundled artifacts from training
- **Multi-entity batch**: Efficient batch predictions
- **Async coordination**: Non-blocking I/O


## Deployment Comparison

### FastAPI (Simple)

**Use Cases:**
- Development/Testing
- Low traffic
- Simple deployment
- No auto-scaling
- No GPU optimization

### Ray Serve (Production)

**Use Cases:**
- Production serving
- High traffic
- Auto-scaling needed
- GPU inference
- Multi-model serving
- Higher complexity

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
