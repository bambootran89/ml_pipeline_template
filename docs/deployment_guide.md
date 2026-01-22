# Zero-to-Hero Deployment Guide

This guide is a comprehensive, step-by-step manual to deploy the `ml-pipeline-template` from scratch locally using Minikube. It is structured into two main scenarios:
1.  **Scenario A: Feast Mode (Advanced)** - Full Feature Store integration (Offline + Online Store).
2.  **Scenario B: Standard Mode (Simple)** - Basic ML pipeline without Feature Store.

## Table of Contents
1.  [Prerequisites & Installation](#1-prerequisites--installation)
2.  [Scenario A: Feast Mode (Advanced)](#2-scenario-a-feast-mode-advanced)
3.  [Scenario B: Standard Mode (Basic)](#3-scenario-b-standard-mode-basic)
4.  [Customizing the Training Pipeline](#4-customizing-the-training-pipeline)
6.  [Updating the Model (Retraining & Reloading)](#6-updating-the-model-retraining--reloading)
7.  [Cleanup](#7-cleanup)

---

## 1. Prerequisites & Installation

You need **Docker**, **kubectl**, and **Minikube** installed.

### Install Docker
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo usermod -aG docker $USER
# Log out and log back in
```

### Install kubectl & Minikube
```bash
# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Start Minikube
```bash
minikube start --driver=docker --cpus=4 --memory=8192
eval $(minikube docker-env) # Point shell to Minikube Docker daemon
```

---

## 2. Scenario A: Feast Mode (Advanced)

Use this mode for full Feature Store integration (Offline + Online Store).

### Step 1: Build & Deploy
Copy and run this entire block to build images and deploy the Feast training job.
```bash
# 1. Build Images
./docker-build.sh -i train
./docker-build.sh -i serve

# 2. Deploy Feast Mode
# Important: Ensure you have initialized the feature store locally first!
# 1. Ingest Data
python -m mlproject.src.pipeline.feature_ops.ingest_batch_etth1 --csv mlproject/data/ETTh1.csv --repo feature_repo_etth1

# 2. Materialize features (Offline -> Online)
python -m mlproject.src.pipeline.feature_ops.materialize_etth1 --repo feature_repo_etth1 --data feature_repo_etth1/data/features.parquet

# 3. Deploy
./deploy.sh -m feast

# 3. Wait for Training to Start
echo "Waiting for training job..."
kubectl wait --for=condition=ready pod -l job-name=training-job-fast-etth1 --timeout=60s
```

### Step 2: Verify Training & Populate Online Store
Run this block to follow the logs and automatically populate the online store once training finishes.
```bash
# 1. Stream Logs (Ctrl+C to exit if needed, but it should finish)
kubectl logs -f job/training-job-fast-etth1

# 2. Populate Online Store (Materialize)
# This assumes the training job has completed successfully.
echo "Materializing Online Store..."
POD_NAME=$(kubectl get pods -l app=ml-prediction -o jsonpath="{.items[0].metadata.name}")
kubectl exec $POD_NAME -- bash -c "cd feature_repo_etth1 && feast materialize-incremental $(date -u +'%Y-%m-%dT%H:%M:%S')"
```

### Step 3: Verify API
Run this block to expose the service and run automated tests.
```bash
# 1. Expose Service (Background)
pkill -f "kubectl port-forward"
kubectl port-forward service/ml-prediction-service 8000:80 > /dev/null 2>&1 &
echo "Service exposed on localhost:8000"

# 2. Run Automated Verification
sleep 5
./test_api.sh
```

---

## 3. Scenario B: Standard Mode (Basic)

Use this mode for simpler pipelines without Feast.

### Step 1: Build & Deploy
```bash
# 1. Build & Deploy
./docker-build.sh -i train
./docker-build.sh -i serve
./deploy.sh -m standard

# 2. Wait for Training
echo "Waiting for training job..."
kubectl wait --for=condition=ready pod -l job-name=training-job-standard-etth1 --timeout=60s
```

### Step 2: Verify Training & API
```bash
# 1. Watch Logs
kubectl logs -f job/training-job-standard-etth1

# 2. Restart API (to load new model)
kubectl rollout restart deployment/ml-prediction-api-standard
kubectl rollout status deployment/ml-prediction-api-standard

# 3. Test
pkill -f "kubectl port-forward"
kubectl port-forward service/ml-prediction-service 8000:80 > /dev/null 2>&1 &
sleep 5
./test_api.sh
```

---

## 4. Manual Verification (Advanced)

If you need to debug specific endpoints manually, use these **fully copy-pasteable** commands.

### Port Forwarding
```bash
pkill -f "kubectl port-forward"
kubectl port-forward service/ml-prediction-service 8000:80 > /dev/null 2>&1 &
```

### 4.1. Health Check
```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### 4.2. Standard Prediction (24-Hour Payload)
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    }
  }' | python3 -m json.tool
```

### 4.3. Batch Predict (Feast Only)
```bash
curl -s -X POST "http://localhost:8000/predict/feast/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "time_point": "2024-01-09T00:00:00",
    "entities": [1, 2]
  }' | python3 -m json.tool
```

### 4.4. Multistep Prediction (Time-Series Only)
```bash
curl -s -X POST "http://localhost:8000/predict/multistep" \
  -H "Content-Type: application/json" \
  -d '{
    "steps_ahead": 12,
    "data": {
      "date": ["2020-01-01 00:00:00", "2020-01-01 01:00:00", "2020-01-01 02:00:00", "2020-01-01 03:00:00", "2020-01-01 04:00:00", "2020-01-01 05:00:00", "2020-01-01 06:00:00", "2020-01-01 07:00:00", "2020-01-01 08:00:00", "2020-01-01 09:00:00", "2020-01-01 10:00:00", "2020-01-01 11:00:00", "2020-01-01 12:00:00", "2020-01-01 13:00:00", "2020-01-01 14:00:00", "2020-01-01 15:00:00", "2020-01-01 16:00:00", "2020-01-01 17:00:00", "2020-01-01 18:00:00", "2020-01-01 19:00:00", "2020-01-01 20:00:00", "2020-01-01 21:00:00", "2020-01-01 22:00:00", "2020-01-01 23:00:00"],
      "HUFL": [5.827, 5.8, 5.969, 6.372, 7.153, 7.976, 8.715, 9.340, 9.763, 9.986, 10.040, 9.916, 9.609, 9.156, 8.591, 7.970, 7.338, 6.745, 6.233, 5.838, 5.582, 5.465, 5.465, 5.557],
      "MUFL": [1.599, 1.492, 1.492, 1.492, 1.492, 1.509, 1.582, 1.711, 1.896, 2.113, 2.337, 2.552, 2.742, 2.902, 3.024, 3.104, 3.137, 3.125, 3.067, 2.969, 2.838, 2.683, 2.515, 2.346],
      "mobility_inflow": [1.234, 1.456, 1.678, 1.890, 2.123, 2.456, 2.789, 3.012, 3.234, 3.456, 3.678, 3.890, 4.012, 4.123, 4.234, 4.345, 4.456, 4.567, 4.678, 4.789, 4.890, 4.901, 4.912, 4.923]
    }
  }' | python3 -m json.tool
```

---

## 5. Customizing the Training Pipeline
You can specify a custom pipeline configuration file located in `mlproject/configs/pipelines/`.

**Syntax:**
```bash
./deploy.sh -m [mode] -p [pipeline_config] -e [experiment_config]
```

**Options:**
- `-m`: Deployment mode (`standard` or `feast`, default: `standard`)
- `-p`: Pipeline config file (default: `standard_train.yaml`)
- `-e`: Experiment config file (default: derived from mode)
- `-n`: Namespace (default: `ml-pipeline`)
- `-d`: Dry run (generate manifests only, do not deploy)
- `-h`: Show help message

**Examples:**

1.  **Feast Mode with Conditional Branching**:
    ```bash
    # Uses pipelines/conditional_branch_feast.yaml AND experiments/etth3_feast.yaml (default)
    ./deploy.sh -m feast -p conditional_branch_feast.yaml
    ```

2.  **Tabular Scenario (Custom Experiment)**:
    ```bash
    # Uses pipelines/conditional_branch_tabular.yaml AND experiments/tabular.yaml
    ./deploy.sh -m feast -p conditional_branch_tabular.yaml -e tabular.yaml
    ```

**How it works:**
The script injects:
*   `{{PIPELINE_CONFIG}}` -> `mlproject/configs/pipelines/...`
*   `{{EXPERIMENT_CONFIG}}` -> `mlproject/configs/experiments/...`
*   Generated manifests are stored in `k8s/generated/` for debugging.

---

## 6. Updating the Model (Retraining & Reloading)

If you have new data or want to update the model with a different configuration:

### 6.1. Retrain the Model
Just run `deploy.sh` again with the same or different configuration. It will automatically delete the old job and start a new one.
```bash
./deploy.sh -m feast -p conditional_branch_feast.yaml
```

### 6.2. Reload the API
Once the training job is **Complete**, you must restart the API deployment to load the new model into memory.
```bash
# For Feast Mode
kubectl rollout restart deployment/ml-prediction-api-feast

# For Standard Mode
kubectl rollout restart deployment/ml-prediction-api-standard
```

---

## 7. Cleanup

To remove all deployed resources, stop port-forwarding, and clean up the environment:

```bash
# 1. Run the cleanup script
chmod +x cleanup.sh
./cleanup.sh

# 2. (Optional) Stop Minikube
minikube stop
```

---

## 8. MLflow Tracking (New)

The deployment now includes a dedicated MLflow Server running in the cluster.

### Accessing the UI
```bash
# Forward port 5000
kubectl port-forward service/mlflow-service 5000:5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### Data Persistence
*   **Artifacts**: Stored in your local project's `mlruns/` folder.
*   **Database**: Stored as `mlruns/mlflow.db` (SQLite).
*   **Why?**: This ensures you can access experiments locally even if you delete the Kubernetes cluster.

---

## 9. Troubleshooting

### Connection Refused / Unable to Connect to Server
*   **Cause**: Minikube is not running or `kubectl` is configured for a different cluster.
*   **Fix**: Run `minikube status`. If stopped, run `minikube start`.

### Minikube fails with PROVIDER_DOCKER_VERSION_EXIT_1
*   **Error**: `failed to connect to the docker API` OR `System has not been booted with systemd`.
*   **Cause**: Docker daemon is not running, often common in WSL2 without systemd enabled.
*   **Fix**:
    1.  Try the SysVinit command (common for WSL):
        ```bash
        sudo service docker start
        ```
    2.  If that doesn't work, try systemd:
        ```bash
        sudo systemctl start docker
        ```
    3.  **WSL2 Users**: Ensure Docker Desktop is running if you are using it.
    4.  Retry `minikube start`.

### ImagePullBackOff / ImageNotFound
*   **Cause**: Kubernetes cannot find your local Docker images because your shell isn't pointing to Minikube's Docker daemon.
*   **Fix**:
    ```bash
    eval $(minikube docker-env)
    ./docker-build.sh -i train
    ./docker-build.sh -i serve
    ```

### HostPath / Mount Errors
*   **Cause**: The script attempts to mount local directories (like `mlruns`). The updated script automatically detects your project root, but ensure you are running `deploy.sh` from the project root directory.
*   **Fix**: Always run scripts from the project root:
    ```bash
    cd /path/to/ml_pipeline_template
    ./deploy.sh ...
    ```
