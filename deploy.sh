#!/bin/bash
# deploy.sh - Automated deployment script for Feast or Standard mode with Dynamic Config

set -e

MODE=${1:-standard}          # Default mode: standard
PIPELINE_CONFIG=${2:-standard_train.yaml} # Default pipeline config

# Default Experiment Config based on Mode
if [ -n "$3" ]; then
    EXPERIMENT_CONFIG="$3"
elif [ "$MODE" == "feast" ]; then
    EXPERIMENT_CONFIG="etth3_feast.yaml"
else
    EXPERIMENT_CONFIG="etth3.yaml"
fi

NAMESPACE="ml-pipeline"

echo "=================================================="
echo " Starting Deployment"

# Derive Serving Config from Pipeline Config
# e.g. standard_train.yaml -> standard_train_serve.yaml
SERVING_CONFIG="${PIPELINE_CONFIG%.yaml}_serve.yaml"

NAMESPACE="ml-pipeline"

echo "=================================================="
echo " Starting Deployment"
echo " Mode:              [$MODE]"
echo " Pipeline Config:   [$PIPELINE_CONFIG]"
echo " Experiment Config: [$EXPERIMENT_CONFIG]"
echo " Serving Config:    [$SERVING_CONFIG]"
echo "=================================================="

# 1. Validation
if [[ "$MODE" != "feast" && "$MODE" != "standard" ]]; then
    echo "Error: Invalid mode '$MODE'. Use 'feast' or 'standard'."
    exit 1
fi

# Check if pipeline config exists (optional strict check)
if [ ! -f "mlproject/configs/pipelines/$PIPELINE_CONFIG" ]; then
    echo "Warning: Config 'mlproject/configs/pipelines/$PIPELINE_CONFIG' not found locally."
fi

# 2. Setup Context
echo "[Setup] Creating namespace $NAMESPACE if not exists..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
kubectl config set-context --current --namespace=$NAMESPACE

# 3. Cleanup Old Resources
echo "[Cleanup] Removing existing jobs and deployments..."
kubectl delete job training-job-fast-etth1 --ignore-not-found
kubectl delete job training-job-standard-etth1 --ignore-not-found
kubectl delete deployment ml-prediction-api-feast --ignore-not-found
kubectl delete deployment ml-prediction-api-standard --ignore-not-found

# 4. Deploy Training Job with Config Injection
echo "[Deploy] Preparing Job Manifest..."
# Create a temporary manifest with the injected config names
sed -e "s/{{PIPELINE_CONFIG}}/$PIPELINE_CONFIG/g" \
    -e "s/{{EXPERIMENT_CONFIG}}/$EXPERIMENT_CONFIG/g" \
    k8s/job-training-$MODE.yaml > k8s/tmp-job-training-$MODE.yaml

echo "[Deploy] Submitting Training Job: k8s/tmp-job-training-$MODE.yaml"
kubectl apply -f k8s/tmp-job-training-$MODE.yaml
rm k8s/tmp-job-training-$MODE.yaml # Cleanup temp file

echo "[Monitor] Waiting for training job to start..."
sleep 5
kubectl get jobs

# 5. Deploy API Service with Config Injection
echo "[Deploy] Preparing API Deployment Manifest..."
sed -e "s/{{PIPELINE_CONFIG}}/$PIPELINE_CONFIG/g" \
    -e "s/{{EXPERIMENT_CONFIG}}/$EXPERIMENT_CONFIG/g" \
    -e "s/{{SERVING_CONFIG}}/$SERVING_CONFIG/g" \
    k8s/deployment-api-$MODE.yaml > k8s/tmp-deployment-api-$MODE.yaml

echo "[Deploy] Submitting API Deployment: k8s/tmp-deployment-api-$MODE.yaml"
kubectl apply -f k8s/tmp-deployment-api-$MODE.yaml
rm k8s/tmp-deployment-api-$MODE.yaml # Cleanup temp file

# 6. Apply Service (Common)
echo "[Deploy] Ensuring Service exists..."
kubectl apply -f k8s/service-api.yaml

echo "=================================================="
echo " Deployment Submitted Successfully!"
echo " Use 'kubectl get pods' to monitor progress."
echo "=================================================="
