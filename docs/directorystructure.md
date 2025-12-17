
# Directory Structure
A layout designed for scalability and feature-store integration.
```plaintext
ml_pipeline_template/
├── .github/                   # CI/CD Workflows (GitHub Actions)
├── k8s/                       # Kubernetes Manifests (Job, Deployment, Service)
├── mlproject/
│   ├── configs/               # Hydra Configurations (The "Control Center")
│   │   ├── base/              # Base configs (model, data, training...)
│   │   └── experiments/       # Specific experiment overrides (e.g., etth1.yaml)
│   ├── data/                  # Raw data storage (git-ignored in prod)
│   ├── serve/                 # Serving Logic
│   │   ├── ray/               # Ray Serve deployment scripts (ray_deploy.py)
│   │   ├── api.py             # FastAPI entrypoint (Local/Docker basic serving)
│   │   ├── models_service.py  # Model loading & inference logic independent of API
│   │   └── schemas.py         # Pydantic schemas for API validation
│   ├── src/
│   │   ├── datamodule/        # Data loading, splitting, and dataset classes
│   │   ├── eval/              # Evaluation metrics & strategies
│   │   ├── models/            # Model definitions (XGBoost, NLinear, TFT...)
│   │   ├── pipeline/          # Orchestrators (Train, Eval, Serve pipelines)
│   │   │   └── engines/       # Execution engines (CV Fold Runner, Tuning Pipeline)
│   │   ├── preprocess/        # Offline & Online data cleaning/scaling
│   │   ├── tracking/          # MLflow wrappers (Experiment, Run, Registry, PyFunc)
│   │   ├── trainer/           # Training loops (ML vs DL differentiation)
│   │   ├── tuning/            # Optuna hyperparameter tuning logic
│   │   └── utils/             # Helper functions (ConfigLoader, MLflowUtils...)
│   ├── run.py                 # Main CLI entrypoint (Training/Prediction)
│   └── __init__.py
├── tests/                     # Unit & Integration Tests (End2End, Ray, Pipeline...)
├── Dockerfile                 # Multi-stage Docker build
├── Makefile                   # Handy commands for dev & ops
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation script
```
