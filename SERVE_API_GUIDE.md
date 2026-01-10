# 🚀 Serve API Auto-Generation & Running Guide

Tự động sinh code API từ serve config và chạy luôn - không cần làm thủ công!

## ⚡ Quick Start (Cách nhanh nhất)

### Cách 1: Bash Script (Recommended)

```bash
# Make script executable (chỉ cần chạy 1 lần)
chmod +x serve_api.sh

# Chạy FastAPI
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# Chạy Ray Serve trên port 9000
./serve_api.sh -f ray -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Cách 2: Python Script

```bash
# FastAPI (default)
python serve_api.py --serve-config mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
python serve_api.py \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --framework ray \
    --port 9000
```

### Cách 3: Python Module

```bash
python -m mlproject.serve.run_generated_api \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --framework fastapi \
    --port 8000
```

---

## 📋 Tất cả các Options

```bash
python serve_api.py \
    --serve-config <path_to_serve.yaml>     # Required: Serve config
    --train-config <path_to_train.yaml>     # Optional: Auto-inferred nếu không có
    --framework <fastapi|ray>               # Optional: Default fastapi
    --host <host>                           # Optional: Default 0.0.0.0
    --port <port>                           # Optional: Default 8000
```

---

## 🎯 Examples

### Example 1: Standard Single Model

```bash
# FastAPI
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
./serve_api.sh -f ray mlproject/configs/generated/standard_train_serve.yaml
```

### Example 2: Conditional Branch (Multi-Model)

```bash
# FastAPI
./serve_api.sh mlproject/configs/generated/conditional_branch_serve.yaml

# Ray Serve với custom port
./serve_api.sh -f ray -p 9000 mlproject/configs/generated/conditional_branch_serve.yaml
```

### Example 3: KMeans + XGBoost Pipeline

```bash
./serve_api.sh mlproject/configs/generated/kmeans_then_xgboost_serve.yaml
```

### Example 4: Custom Host & Port

```bash
python serve_api.py \
    --serve-config mlproject/configs/generated/standard_train_serve.yaml \
    --host 127.0.0.1 \
    --port 5000
```

---

## 🔧 Quy trình tự động

Khi chạy script, nó sẽ tự động:

1. **Generate API code** từ serve.yaml
2. **Configure** host và port
3. **Run** API server ngay lập tức

```
[1/3] Generating API code...
✓ Generated: mlproject/serve/generated/standard_train_serve_fastapi.py

[2/3] Configuring server settings...
✓ Configured: 0.0.0.0:8000

[3/3] Starting FASTAPI server...

============================================================
🚀 API is starting at: http://0.0.0.0:8000
📖 API docs: http://0.0.0.0:8000/docs
❤️  Health check: http://0.0.0.0:8000/health
============================================================

💡 Press Ctrl+C to stop the server
```

---

## 📊 API Endpoints

Sau khi server chạy:

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "feature1": [1.0, 2.0, 3.0],
      "feature2": [4.0, 5.0, 6.0]
    }
  }'
```

Response:
```json
{
  "predictions": [0.123, 0.456, 0.789]
}
```

### Interactive Docs (FastAPI only)
Mở browser: `http://localhost:8000/docs`

---

## 🎨 Framework Comparison

### FastAPI
✅ **Pros:**
- Lightweight, nhanh
- Auto-generated docs (Swagger UI)
- Dễ debug
- Synchronous (đơn giản)

❌ **Cons:**
- Single process (không scale tự động)
- Phải dùng load balancer để scale

**Best for:** Development, small deployments, single-model serving

### Ray Serve
✅ **Pros:**
- Distributed, scale tự động
- Multi-replica (load balancing built-in)
- Async processing
- Dashboard để monitor
- Production-ready

❌ **Cons:**
- Phức tạp hơn
- Tốn resource hơn
- Setup phức tạp hơn

**Best for:** Production, high-traffic, multi-model serving

---

## 🛠️ Troubleshooting

### Port already in use
```bash
# Dùng port khác
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Module not found
```bash
# Đảm bảo đang ở thư mục root
cd /home/user/ml_pipeline_template

# Hoặc set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Model not loading
- Check MLflow tracking URI: `echo $MLFLOW_TRACKING_URI`
- Verify model exists: `mlflow models list`
- Check alias: Mặc định là "production"

### Import errors
```bash
# Install dependencies
pip install fastapi uvicorn ray[serve]
```

---

## 📁 Generated Files Location

Tất cả generated files được lưu trong:
```
mlproject/serve/generated/
├── standard_train_serve_fastapi.py
├── standard_train_serve_ray.py
├── conditional_branch_serve_fastapi.py
├── conditional_branch_serve_ray.py
└── ...
```

Bạn có thể:
- ✅ Chỉnh sửa để customize
- ✅ Check vào git để track changes
- ✅ Deploy trực tiếp lên production
- ✅ Regenerate bất cứ lúc nào

---

## 🔄 Workflow Complete

### Development Flow
```bash
# 1. Train model
python -m mlproject.src.pipeline.dag_run train ...

# 2. Generate serve config
python -m mlproject.src.pipeline.dag_run generate ...

# 3. Run API (TỰ ĐỘNG SINH CODE!)
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# 4. Test
curl http://localhost:8000/health
```

### One-Liner Development
```bash
# Generate và run ngay
python -m mlproject.src.pipeline.dag_run generate \
    mlproject/configs/pipelines/standard_train.yaml \
    --config-type serve \
    --output-dir mlproject/configs/generated \
  && ./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

---

## 💡 Pro Tips

### 1. Chạy nhiều APIs cùng lúc (different ports)
```bash
# Terminal 1: Model A
./serve_api.sh -p 8000 mlproject/configs/generated/standard_train_serve.yaml

# Terminal 2: Model B
./serve_api.sh -p 8001 mlproject/configs/generated/conditional_branch_serve.yaml
```

### 2. Background running
```bash
# FastAPI với nohup
nohup ./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml > api.log 2>&1 &

# Check log
tail -f api.log
```

### 3. Docker deployment
```dockerfile
FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Run API on container start
CMD ["python", "serve_api.py", \
     "--serve-config", "mlproject/configs/generated/standard_train_serve.yaml", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

### 4. Kubernetes deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: your-ml-api:latest
        command:
          - python
          - serve_api.py
          - --serve-config
          - mlproject/configs/generated/standard_train_serve.yaml
        ports:
        - containerPort: 8000
```

---

## 📚 Related Documentation

- **API Generation**: `mlproject/serve/generated/README.md`
- **Config Generation**: `mlproject/src/utils/generator/README.md` (if exists)
- **Example Scripts**: `examples/generate_serve_apis.py`

---

## 🎓 Summary

**Cách dùng đơn giản nhất:**
```bash
./serve_api.sh <serve_config.yaml>
```

**That's it!** 🎉

Script sẽ:
1. ✅ Tự động generate code
2. ✅ Tự động configure
3. ✅ Tự động run server
4. ✅ Hiển thị URLs để test

**Không cần làm gì thủ công nữa!**
