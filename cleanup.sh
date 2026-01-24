#!/bin/bash
# cleanup.sh - Complete cleanup of ML Pipeline Kubernetes resources

set -e

NAMESPACE="ml-pipeline"

echo "=================================================="
echo " Cleaning up ML Pipeline Resources"
echo "=================================================="

# 1. Stop Port Forwarding
echo "[1/4] Stopping port-forwarding processes..."
pkill -f "kubectl port-forward" || true

# 2. Delete Jobs and Deployments
echo "[2/4] Deleting Jobs and Deployments in namespace '$NAMESPACE'..."
kubectl delete jobs --all -n $NAMESPACE --ignore-not-found
kubectl delete deployments --all -n $NAMESPACE --ignore-not-found
kubectl delete services --all -n $NAMESPACE --ignore-not-found

# 3. Optional: Remove Namespace
# echo "[3/4] Removing namespace '$NAMESPACE'..."
# kubectl delete namespace $NAMESPACE --ignore-not-found

# 4. Cleanup MLflow and Local Temporary Manifests
# 4. Local Resource Cleanup (Ray & Ports)
echo "[4/5] Cleaning up Local Resources..."
echo " - Stopping Ray..."
ray stop > /dev/null 2>&1 || true

echo " - Freeing port 8082..."
fuser -k 8082/tcp > /dev/null 2>&1 || true
if lsof -t -i:8082 > /dev/null 2>&1; then
    lsof -t -i:8082 | xargs kill -9 > /dev/null 2>&1 || true
fi

# 5. Cleanup MLflow and generated files
echo "[5/5] Cleaning up MLflow artifacts..."
kubectl delete deployment mlflow-server -n $NAMESPACE --ignore-not-found || true
kubectl delete service mlflow-service -n $NAMESPACE --ignore-not-found || true
# Optional: Clear temp logs if desired, uncomment to valid
# rm -rf tmplogs/

echo "=================================================="
echo " Cleanup Complete!"
echo "=================================================="
