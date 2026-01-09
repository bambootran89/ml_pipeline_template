# Hướng dẫn sử dụng ConfigGenerator với Nested Sub-Pipelines

## Tổng quan

`ConfigGenerator` đã được nâng cấp để hỗ trợ tự động generate các config eval/serve từ training pipeline có chứa **nested sub-pipelines**.

### Các loại pipeline được hỗ trợ:
- ✅ Standard flat pipelines (`standard_train.yaml`)
- ✅ **Nested sub-pipelines (`nested_suppipeline.yaml`)** ← MỚI
- ✅ Two-stage pipelines (`kmeans_then_xgboost.yaml`)
- ✅ Parallel ensemble pipelines

## Cách sử dụng

### 1. Generate cả Eval và Serve configs (khuyến nghị)

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest
```

**Output:**
- `mlproject/configs/generated/nested_suppipeline_eval.yaml`
- `mlproject/configs/generated/nested_suppipeline_serve.yaml`

### 2. Generate chỉ Eval config

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest \
    --type eval
```

### 3. Generate chỉ Serve config

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest \
    --type serve
```

## Sử dụng configs đã generate

### Chạy Evaluation

```bash
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_eval.yaml \
    -a latest
```

### Chạy Serving

```bash
# Với CSV input
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_serve.yaml \
    -i ./sample_input.csv \
    -a latest

# Với Feast Feature Store
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_serve.yaml \
    -a latest \
    --time_point "2024-01-01 12:00:00"
```

## Kiến trúc Transformation

### Training Config → Eval Config

**Input (Training):**
```yaml
steps:
  - id: load_data
    type: data_loader

  - id: feature_pipeline
    type: sub_pipeline
    depends_on: [load_data]
    pipeline:
      steps:
        - id: normalize
          type: preprocessor
          is_train: true
          log_artifact: true

        - id: cluster
          type: clustering
          log_artifact: true
          hyperparams:
            n_clusters: 4

  - id: train_model
    type: trainer
    log_artifact: true
```

**Output (Eval):**
```yaml
steps:
  - id: load_data
    type: data_loader

  - id: init_artifacts
    type: mlflow_loader
    load_map:
      - {step_id: normalize, context_key: normalize_model}
      - {step_id: cluster, context_key: cluster_model}
      - {step_id: train_model, context_key: train_model_model}

  - id: feature_pipeline
    type: sub_pipeline
    depends_on: [load_data]  # ← Giữ nguyên
    pipeline:
      steps:
        - id: normalize
          type: preprocessor
          is_train: false      # ← Load mode
          alias: latest
          instance_key: normalize_model

        - id: cluster
          type: clustering
          wiring:
            inputs:
              model: cluster_model  # ← Load từ MLflow

  - id: cluster_evaluate      # ← Auto-generated
    type: evaluator

  - id: train_evaluate        # ← Auto-generated
    type: evaluator
```

### Training Config → Serve Config

**Output (Serve):**
```yaml
steps:
  - id: init_artifacts
    type: mlflow_loader
    load_map:
      - {step_id: normalize, context_key: normalize_model}
      - {step_id: cluster, context_key: cluster_model}
      - {step_id: train_model, context_key: train_model_model}

  - id: feature_pipeline
    type: sub_pipeline
    # ← KHÔNG có depends_on (data được pre-init)
    pipeline:
      steps:
        - id: normalize
          type: preprocessor
          is_train: false
          alias: latest
          instance_key: normalize_model

        - id: cluster
          type: clustering
          wiring:
            inputs:
              model: cluster_model

  - id: train_model_inference  # ← Auto-generated
    type: inference
```

## Điểm quan trọng

### 1. **Eval Mode vs Serve Mode**

| Aspect | Eval Mode | Serve Mode |
|--------|-----------|------------|
| Data Loading | `load_data` step | Pre-initialized qua `prepare_serving_data()` |
| Sub-pipeline deps | `depends_on: [load_data]` | Không có `depends_on` |
| Model steps | → `evaluator` | → `inference` |

### 2. **Recursive Extraction**

ConfigGenerator tự động:
- Trích xuất tất cả model producers (trainer, clustering) từ nested structures
- Trích xuất tất cả preprocessors từ nested structures
- Build load_map đầy đủ cho MLflow loader

### 3. **Sub-pipeline Transformation**

Sub-pipelines được giữ nguyên structure nhưng các internal steps được transform:

**Preprocessor steps:**
- `is_train: true` → `is_train: false`
- Thêm `alias` và `instance_key`
- Remove `log_artifact`, `artifact_type`
- Remove training-only wiring inputs

**Clustering steps:**
- Thêm wiring để load model từ MLflow
- Remove `hyperparams`
- Remove `log_artifact`, `artifact_type`

## Ví dụ thực tế

### Standard Training Pipeline
```bash
# 1. Generate configs
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/standard_train.yaml \
    -o mlproject/configs/generated \
    -a latest

# 2. Chạy eval
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/standard_train_eval.yaml \
    -a latest
```

### Nested Sub-Pipeline
```bash
# 1. Generate configs
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest

# 2. Chạy eval
python -m mlproject.src.pipeline.dag_run eval \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_eval.yaml \
    -a latest

# 3. Chạy serve
python -m mlproject.src.pipeline.dag_run serve \
    -e mlproject/configs/experiments/etth3.yaml \
    -p mlproject/configs/generated/nested_suppipeline_serve.yaml \
    -i ./sample_input.csv \
    -a latest
```

## Python API

Nếu bạn muốn sử dụng trực tiếp trong Python:

```python
from mlproject.src.utils.config_generator import ConfigGenerator

# Initialize
generator = ConfigGenerator("mlproject/configs/pipelines/nested_suppipeline.yaml")

# Generate all configs
results = generator.generate_all(
    output_dir="mlproject/configs/generated",
    alias="production"
)

print(f"Eval config: {results['eval']}")
print(f"Serve config: {results['serve']}")

# Or generate individually
generator.generate_eval_config(
    alias="latest",
    output_path="path/to/eval_config.yaml"
)

generator.generate_serve_config(
    alias="staging",
    output_path="path/to/serve_config.yaml"
)
```

## Troubleshooting

### Issue 1: "Sub-pipeline depends_on load_data but load_data not found"

**Giải pháp:** Regenerate serve config với version mới nhất của ConfigGenerator.

```bash
python -m mlproject.src.pipeline.dag_run generate \
    -t mlproject/configs/pipelines/nested_suppipeline.yaml \
    -o mlproject/configs/generated \
    -a latest \
    --type serve
```

### Issue 2: "Model not found in context"

**Kiểm tra:**
1. MLflow loader có load đúng artifacts không?
2. `instance_key` trong sub-pipeline steps có match với `context_key` trong load_map không?

```yaml
# load_map nên có:
load_map:
  - step_id: normalize
    context_key: normalize_model  # ← Match với instance_key

# Sub-pipeline step nên có:
- id: normalize
  instance_key: normalize_model  # ← Match với context_key
```

### Issue 3: "Data not found in serve mode"

Serving mode sử dụng `prepare_serving_data()` để pre-initialize context với data.
Không cần `load_data` step trong serve config.

## Tài liệu tham khảo

- Source code: `mlproject/src/utils/config_generator.py`
- Examples: `mlproject/examples/`
- Test configs: `mlproject/configs/generated/`
- Pipeline configs: `mlproject/configs/pipelines/`

## Changelog

### v2.0 (Current)
- ✅ Added nested sub-pipeline support
- ✅ Recursive extraction of model producers and preprocessors
- ✅ Fixed serve mode to not depend on load_data
- ✅ Maintained backward compatibility with flat pipelines

### v1.0
- Initial support for standard flat pipelines
- Support for two-stage pipelines (kmeans_then_xgboost)
