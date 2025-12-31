# Directory Structure

## Overview

The project structure is designed for **scalability, modularity, and feature-store integration**. Each directory has a clear responsibility following **separation of concerns** principle.


## Complete Directory Tree

```text
mlproject/
│
├── configs/                     # Hydra configuration hub
│   ├── base/                    # Base configs (shared)
│   │   ├── model.yaml
│   │   ├── training.yaml
│   │   ├── preprocessing.yaml
│   │   ├── evaluation.yaml
│   │   ├── mlflow.yaml
│   │   └── tuning.yaml
│   └── experiments/             # Experiment-specific overrides
│       ├── etth1.yaml
│       ├── etth1_feast.yaml
│       ├── etth2_tuning.yaml
│       └── titanic_feast.yaml
│
├── data/                        # Raw data (git-ignored)
│   ├── ETTh1.csv
│   ├── titanic.csv
│   └── README.md
│
├── artifacts/                    # Generated artifacts (git-ignored)
│   ├── models/
│   │   ├── model.pkl
│   │   └── model.pt
│   └── preprocessing/
│       ├── fillna_stats.pkl
│       ├── label_encoders.pkl
│       └── scaler.pkl
│
├── serve/                        # Serving layer
│   ├── api.py                    # FastAPI app
│   ├── ray_deploy.py             # Ray Serve deployment
│   ├── models_service.py         # Inference logic
│   └── schemas.py                # Request/response schemas
│
├── src/                          # Core ML logic
│   ├── datamodule/               # Data loading & splitting
│   ├── eval/                     # Evaluation metrics & runners
│   ├── features/                 # Feature store (Feast)
│   ├── models/                   # Model implementations
│   ├── pipeline/                 # Training, evaluation, and serving pipelines
│   ├── preprocess/               # Data transformation & preprocessing
│   ├── tracking/                 # MLflow integration
│   ├── trainer/                  # Training loops (ML & DL)
│   ├── tuning/                   # Hyperparameter tuning (Optuna)
│   └── utils/                    # Helper functions
│
├── tests/                        # Test suite
│   ├── test_datamodule.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── ...
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── offlineworkflow.md
│   ├── onlineworkflow.md
│   └── preprocessing.md
│
├── Docker & Deployment
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
├── Project Files
│   ├── requirements/prod.txt
│   ├── requirements/dev.txt
│   ├── setup.py
│   ├── pyproject.toml
│   ├── Makefile
│   ├── .gitignore
│   └── README.md
│
└── Generated runtime
    ├── mlruns/                   # MLflow tracking
    └── .ray/                     # Ray runtime files
```
