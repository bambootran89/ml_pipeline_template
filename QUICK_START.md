# 🚀 QUICK START - Serve API in 1 Command

## Cách chạy nhanh nhất (1 dòng lệnh):

```bash
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

**Done!** API đã chạy tại `http://localhost:8000` 🎉

---

## Chi tiết:

### ✅ Điều kiện cần:
1. Đã có file serve config (ví dụ: `standard_train_serve.yaml`)
2. Script có quyền executable: `chmod +x serve_api.sh`

### 🎯 Làm gì nếu chưa có serve config?

```bash
# Generate serve config từ training config
python -m mlproject.src.pipeline.dag_run generate \
    mlproject/configs/pipelines/standard_train.yaml \
    --config-type serve \
    --output-dir mlproject/configs/generated
```

### 📋 Các options:

```bash
# FastAPI (default)
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml

# Ray Serve
./serve_api.sh -f ray mlproject/configs/generated/standard_train_serve.yaml

# Custom port
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml

# All options
./serve_api.sh -f ray -p 9000 -h 127.0.0.1 mlproject/configs/generated/standard_train_serve.yaml
```

---

## 🧪 Test API:

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"feature1": [1,2,3], "feature2": [4,5,6]}}'

# Swagger UI (FastAPI only)
# Mở browser: http://localhost:8000/docs
```

---

## 📚 Chi tiết hơn?

Xem file: `SERVE_API_GUIDE.md`

---

## 🔥 Examples:

### Example 1: Standard Pipeline
```bash
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

### Example 2: Conditional Branch (Multi-model)
```bash
./serve_api.sh mlproject/configs/generated/conditional_branch_serve.yaml
```

### Example 3: Ray Serve on Port 9000
```bash
./serve_api.sh -f ray -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

---

## 💡 Behind the scenes:

Khi chạy `./serve_api.sh`, script sẽ:

1. ✅ Tự động sinh code FastAPI/Ray Serve từ serve.yaml
2. ✅ Configure host & port
3. ✅ Run API server

**Bạn không cần làm gì thêm!**

---

## ❓ Troubleshooting:

### Port đã được dùng?
```bash
./serve_api.sh -p 9000 mlproject/configs/generated/standard_train_serve.yaml
```

### Module not found?
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
./serve_api.sh mlproject/configs/generated/standard_train_serve.yaml
```

### Script không chạy được?
```bash
# Make sure executable
chmod +x serve_api.sh

# Or use Python directly
python serve_api.py --serve-config mlproject/configs/generated/standard_train_serve.yaml
```

---

That's it! 🎊
