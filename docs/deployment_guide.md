# Deployment Guide

This guide covers containerization with Docker and deployment to Kubernetes, including local testing with Minikube or Kind.

## Table of Contents

- [Docker Containerization](#docker-containerization)
- [Local Kubernetes Setup](#local-kubernetes-setup)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Production Considerations](#production-considerations)
- [Troubleshooting](#troubleshooting)

---

## Docker Containerization

### Building the Docker Image

The project includes a production-ready Dockerfile with multi-stage optimization.

```bash
# Build the Docker image
docker build -t ml-pipeline-template:latest .

# Build with specific version tag
docker build -t ml-pipeline-template:v1.0.0 .

# Build for specific platform (for M1/M2 Macs deploying to x86)
docker build --platform linux/amd64 -t ml-pipeline-template:latest .
```

### Testing the Docker Image Locally

**Run API server:**

```bash
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  ml-pipeline-template:latest
```

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

**Run training job:**

```bash
docker run --rm \
  --name ml-training \
  -v $(pwd)/mlruns:/app/mlruns \
  -e MLFLOW_TRACKING_URI=file:///app/mlruns \
  ml-pipeline-template:latest \
  python -m mlproject.src.pipeline.dag_run train \
    -e /app/mlproject/configs/experiments/etth3.yaml \
    -p /app/mlproject/configs/pipelines/standard_train.yaml
```

**Run with shell for debugging:**

```bash
docker run -it --rm ml-pipeline-template:latest /bin/bash
```

### Pushing to Container Registry

**Docker Hub:**

```bash
# Login
docker login

# Tag image
docker tag ml-pipeline-template:latest yourusername/ml-pipeline-template:latest

# Push
docker push yourusername/ml-pipeline-template:latest
```

**AWS ECR:**

```bash
# Login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag
docker tag ml-pipeline-template:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline-template:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-pipeline-template:latest
```

**Google Container Registry:**

```bash
# Configure Docker
gcloud auth configure-docker

# Tag
docker tag ml-pipeline-template:latest gcr.io/your-project-id/ml-pipeline-template:latest

# Push
docker push gcr.io/your-project-id/ml-pipeline-template:latest
```

---

## Local Kubernetes Setup

### Option 1: Minikube (Recommended for Mac/Windows)

**Install Minikube:**

```bash
# macOS (Homebrew)
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (Chocolatey)
choco install minikube
```

**Start Minikube cluster:**

```bash
# Start with Docker driver (recommended)
minikube start --driver=docker --cpus=4 --memory=8192

# Start with specific Kubernetes version
minikube start --driver=docker --kubernetes-version=v1.28.0

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

**Load Docker image into Minikube:**

```bash
# Option 1: Use Minikube's Docker daemon
eval $(minikube docker-env)
docker build -t ml-pipeline-template:latest .

# Option 2: Load pre-built image
minikube image load ml-pipeline-template:latest

# Verify image is loaded
minikube image ls | grep ml-pipeline-template
```

**Access services:**

```bash
# Get service URL
minikube service ml-prediction-service --url

# Port forward to localhost
kubectl port-forward service/ml-prediction-service 8000:80

# Open dashboard
minikube dashboard
```

**Stop and cleanup:**

```bash
# Stop cluster
minikube stop

# Delete cluster
minikube delete

# Delete all clusters
minikube delete --all
```

### Option 2: Kind (Kubernetes in Docker)

**Install Kind:**

```bash
# macOS/Linux (Homebrew)
brew install kind

# Linux (direct download)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Windows (Chocolatey)
choco install kind
```

**Create Kind cluster:**

```bash
# Create cluster with custom config
cat <<EOF | kind create cluster --name ml-cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 30000
    hostPort: 8000
    protocol: TCP
- role: worker
- role: worker
EOF

# Simple single-node cluster
kind create cluster --name ml-cluster
```

**Load Docker image into Kind:**

```bash
# Build and load image
docker build -t ml-pipeline-template:latest .
kind load docker-image ml-pipeline-template:latest --name ml-cluster

# Verify image is loaded
docker exec -it ml-cluster-control-plane crictl images | grep ml-pipeline-template
```

**Access services:**

```bash
# Port forward
kubectl port-forward service/ml-prediction-service 8000:80

# For NodePort services (if using extraPortMappings)
# Access via http://localhost:8000
```

**Delete cluster:**

```bash
kind delete cluster --name ml-cluster
```

### Option 3: k3d (Lightweight K3s in Docker)

**Install k3d:**

```bash
# macOS/Linux
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Or via Homebrew
brew install k3d
```

**Create k3d cluster:**

```bash
# Create cluster with port mapping
k3d cluster create ml-cluster \
  --port 8000:80@loadbalancer \
  --agents 2

# With registry
k3d cluster create ml-cluster \
  --port 8000:80@loadbalancer \
  --registry-create ml-registry:0.0.0.0:5000
```

**Load image:**

```bash
# Import image
k3d image import ml-pipeline-template:latest -c ml-cluster

# Or push to local registry
docker tag ml-pipeline-template:latest localhost:5000/ml-pipeline-template:latest
docker push localhost:5000/ml-pipeline-template:latest
```

**Delete cluster:**

```bash
k3d cluster delete ml-cluster
```

### Comparison

| Feature | Minikube | Kind | k3d |
|---------|----------|------|-----|
| **Installation** | Single binary | Single binary | Single binary |
| **Resource Usage** | Medium | Low | Very Low |
| **Startup Time** | 1-2 min | 30-60 sec | 20-30 sec |
| **Multi-node** | Yes (complex) | Yes (easy) | Yes (easy) |
| **LoadBalancer** | Via tunnel | Port mapping | Built-in |
| **Best For** | Local dev, testing | CI/CD, testing | Fast iteration |

---

## Kubernetes Deployment

### Prerequisites

```bash
# Verify cluster is running
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace ml-pipeline

# Set default namespace
kubectl config set-context --current --namespace=ml-pipeline
```

### Deploy MLflow Server (Optional)

```bash
# Create MLflow deployment
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: ghcr.io/mlflow/mlflow:v2.9.2
        ports:
        - containerPort: 5000
        args:
        - server
        - --host=0.0.0.0
        - --port=5000
        - --backend-store-uri=sqlite:///mlflow/mlflow.db
        - --default-artifact-root=/mlflow/artifacts
        volumeMounts:
        - name: mlflow-data
          mountPath: /mlflow
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
spec:
  type: ClusterIP
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
```

### Deploy ConfigMap for Experiments

```bash
# Create ConfigMap from experiment configs
kubectl create configmap ml-configs \
  --from-file=mlproject/configs/experiments/etth3.yaml \
  --from-file=mlproject/configs/pipelines/standard_train.yaml

# Or from directory
kubectl create configmap ml-configs \
  --from-file=mlproject/configs/experiments/
```

### Deploy API Service

```bash
# Apply deployment and service
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/service-api.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/ml-prediction-api

# Scale deployment
kubectl scale deployment ml-prediction-api --replicas=3
```

### Run Training Jobs

```bash
# Apply training job
kubectl apply -f k8s/job-training.yaml

# Check job status
kubectl get jobs
kubectl get pods

# View logs
kubectl logs -f job/training-job-etth1

# Check job completion
kubectl wait --for=condition=complete --timeout=600s job/training-job-etth1

# Delete completed job
kubectl delete job training-job-etth1
```

### Expose API Externally

**Option 1: Port Forward (Testing)**

```bash
kubectl port-forward service/ml-prediction-service 8000:80
# Access via http://localhost:8000
```

**Option 2: NodePort**

```bash
# Update service to NodePort
kubectl patch service ml-prediction-service -p '{"spec":{"type":"NodePort"}}'

# Get NodePort
kubectl get service ml-prediction-service

# Access via http://<node-ip>:<node-port>
# For Minikube
minikube service ml-prediction-service --url
```

**Option 3: LoadBalancer (Cloud)**

```bash
# Update service to LoadBalancer
kubectl patch service ml-prediction-service -p '{"spec":{"type":"LoadBalancer"}}'

# Get external IP (may take a few minutes)
kubectl get service ml-prediction-service -w
```

**Option 4: Ingress**

```bash
# Create Ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  ingressClassName: nginx
  rules:
  - host: ml-api.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-prediction-service
            port:
              number: 80
EOF

# For Minikube, get Ingress IP
minikube ip

# Add to /etc/hosts
echo "$(minikube ip) ml-api.local" | sudo tee -a /etc/hosts

# Access via http://ml-api.local
```

### Health Checks and Monitoring

```bash
# Check pod health
kubectl describe pod <pod-name>

# View resource usage
kubectl top pods
kubectl top nodes

# View events
kubectl get events --sort-by='.lastTimestamp'

# Execute command in pod
kubectl exec -it <pod-name> -- /bin/bash

# Copy files from pod
kubectl cp <pod-name>:/app/mlruns ./local-mlruns
```

---

## Production Considerations

### Resource Management

Update `k8s/deployment-api.yaml` with appropriate resource limits:

```yaml
resources:
  requests:
    cpu: "1000m"      # 1 CPU core minimum
    memory: "2Gi"     # 2GB RAM minimum
  limits:
    cpu: "2000m"      # 2 CPU cores maximum
    memory: "4Gi"     # 4GB RAM maximum
```

### Autoscaling

```bash
# Create Horizontal Pod Autoscaler
kubectl autoscale deployment ml-prediction-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10

# Check HPA status
kubectl get hpa

# Detailed HPA info
kubectl describe hpa ml-prediction-api
```

### Secrets Management

```bash
# Create secret for MLflow credentials
kubectl create secret generic mlflow-secrets \
  --from-literal=aws-access-key-id=YOUR_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET

# Use in deployment
# env:
# - name: AWS_ACCESS_KEY_ID
#   valueFrom:
#     secretKeyRef:
#       name: mlflow-secrets
#       key: aws-access-key-id
```

### Persistent Storage

Update `k8s/job-training.yaml` to use PersistentVolumeClaim:

```yaml
volumes:
- name: model-storage
  persistentVolumeClaim:
    claimName: ml-artifacts-pvc
```

Create PVC:

```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-artifacts-pvc
spec:
  accessModes:
  - ReadWriteMany  # For shared access across pods
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard  # Or your storage class
EOF
```

### Monitoring

Install Prometheus and Grafana:

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus operator
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default credentials: admin/prom-operator
```

### Logging

Collect logs with Fluentd/Elasticsearch/Kibana:

```bash
# Install EFK stack
kubectl apply -f https://raw.githubusercontent.com/fluent/fluentd-kubernetes-daemonset/master/fluentd-daemonset-elasticsearch.yaml

# Or use Loki for lightweight logging
helm install loki grafana/loki-stack \
  --namespace logging \
  --create-namespace
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>

# Common issues:
# - ImagePullBackOff: Image not found or private
#   Solution: Use correct image path, add imagePullSecrets
# - CrashLoopBackOff: Application crashing
#   Solution: Check logs with kubectl logs <pod-name>
# - Pending: Insufficient resources
#   Solution: Check with kubectl describe node
```

### Image Pull Errors

```bash
# Verify image exists locally (for Minikube/Kind)
minikube image ls | grep ml-pipeline-template
# or
docker exec -it kind-control-plane crictl images

# Reload image
minikube image load ml-pipeline-template:latest
# or
kind load docker-image ml-pipeline-template:latest
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints ml-prediction-service

# Check pod labels match service selector
kubectl get pods --show-labels
kubectl describe service ml-prediction-service

# Test from within cluster
kubectl run test-pod --rm -it --image=curlimages/curl -- sh
# Inside pod:
curl http://ml-prediction-service
```

### Training Job Fails

```bash
# Check job status
kubectl describe job training-job-etth1

# View logs
kubectl logs -f job/training-job-etth1

# Common issues:
# - Config file not found: Mount ConfigMap
# - MLflow connection error: Check MLFLOW_TRACKING_URI
# - Out of memory: Increase resource limits
```

### Clean Up Resources

```bash
# Delete all resources in namespace
kubectl delete all --all -n ml-pipeline

# Delete namespace
kubectl delete namespace ml-pipeline

# Delete PVCs
kubectl delete pvc --all -n ml-pipeline
```

---

## Complete Local Testing Example

```bash
# 1. Start Minikube
minikube start --driver=docker --cpus=4 --memory=8192

# 2. Build and load image
eval $(minikube docker-env)
docker build -t ml-pipeline-template:latest .

# 3. Create namespace
kubectl create namespace ml-pipeline
kubectl config set-context --current --namespace=ml-pipeline

# 4. Deploy API
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/service-api.yaml

# 5. Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/ml-prediction-api

# 6. Expose service
kubectl port-forward service/ml-prediction-service 8000:80 &

# 7. Test API
curl http://localhost:8000/health

# 8. Run training job
kubectl apply -f k8s/job-training.yaml

# 9. Monitor job
kubectl logs -f job/training-job-etth1

# 10. Cleanup
kubectl delete all --all
minikube stop
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t ml-pipeline-template:${{ github.sha }} .

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker tag ml-pipeline-template:${{ github.sha }} yourusername/ml-pipeline-template:latest
          docker push yourusername/ml-pipeline-template:latest

      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v1
        with:
          manifests: |
            k8s/deployment-api.yaml
            k8s/service-api.yaml
          images: yourusername/ml-pipeline-template:latest
```

---

## Related Documentation

- [API Generation Guide](./api_generation_guide.md)
- [Pipeline Orchestration](./pipeline_orchestration.md)
- [Verification Guide](./verification_guide.md)
- [Architecture](./architecture.md)
