# Directory Structure

## Overview

The project structure is designed for **scalability, modularity, and feature-store integration**. Each directory has a clear responsibility following **separation of concerns** principle.


## Complete Directory Tree

```
ml_pipeline_template/
│
├── .github/                             # CI/CD & Automation
│   ├── workflows/
│   │   ├── ci.yml                       # Linting, testing
│   │   ├── docker-build.yml             # Docker image build
│   │   └── deploy.yml                   # Deployment automation
│   └── dependabot.yml                   # Dependency updates
│
├── k8s/                                 # Kubernetes Manifests
│   ├── job-training.yaml                # Training job
│   ├── deployment-api.yaml              # API deployment
│   ├── service-api.yaml                 # LoadBalancer service
│   ├── configmap.yaml                   # Config injection
│   └── secrets.yaml                     # Credentials (gitignored)
│
├── mlproject/
│   │
│   ├── configs/                         # Hydra Configuration Hub
│   │   │
│   │   ├── base/                        # Base configs (inherited)
│   │   │   ├── model.yaml               # Model registry
│   │   │   ├── training.yaml            # Training params
│   │   │   ├── preprocessing.yaml       # Transform pipeline
│   │   │   ├── evaluation.yaml          # Metrics config
│   │   │   ├── mlflow.yaml              # MLflow settings
│   │   │   └── tuning.yaml              # Optuna search spaces
│   │   │
│   │   └── experiments/                 # Experiment-specific overrides
│   │       ├── etth1.yaml               # ETTh1 with CSV
│   │       ├── etth1_feast.yaml         # ETTh1 with Feast
│   │       ├── etth2_tuning.yaml        # Hyperparameter tuning
│   │       └── titanic_feast.yaml       # Tabular example
│   │
│   ├── data/                            # Raw Data (Git-ignored)
│   │   ├── ETTh1.csv
│   │   ├── titanic.csv
│   │   └── README.md                    # Data description
│   │
│   ├── serve/                           # Serving Layer
│   │   ├── api.py                       # FastAPI app (simple)
│   │   ├── ray_deploy.py                # Ray Serve (production)
│   │   ├── models_service.py            # Inference logic
│   │   └── schemas.py                   # Request/Response schemas
│   │
│   ├── src/                             # Core ML Logic
│   │   │
│   │   ├── datamodule/                  # Data Loading
│   │   │   ├── base.py                  # BaseDataModule interface
│   │   │   ├── ml_datamodule.py         # Tabular data
│   │   │   ├── ts_datamodule.py         # Time-series data
│   │   │   ├── loaders/
│   │   │   │   ├── base.py              # DataLoader interface
│   │   │   │   ├── feast_loader.py      # Feast integration
│   │   │   │   ├── file_loader.py       # CSV/Parquet
│   │   │   │   └── factory.py           # DataLoaderFactory
│   │   │   └── splitter.py              # Train/Val/Test split
│   │   │
│   │   ├── eval/                        # Evaluation Metrics
│   │   │   ├── metrics.py               # MAE, RMSE, MAPE
│   │   │   └── evaluator.py             # Evaluation runner
│   │   │
│   │   ├── features/                    # Feature Store (Feast)
│   │   │   ├── facade.py                # FeatureStoreFacade
│   │   │   ├── factory.py               # FeatureStoreFactory
│   │   │   ├── strategies/              # Strategy pattern
│   │   │   │   ├── base.py              # Strategy interface
│   │   │   │   ├── timeseries.py        # Timeseries retrieval
│   │   │   │   ├── tabular.py           # Tabular retrieval
│   │   │   │   ├── online.py            # Online retrieval
│   │   │   │   └── factory.py           # StrategyFactory
│   │   │   └── timeseries.py            # TimeSeriesFeatureStore
│   │   │
│   │   ├── models/                      # Model Implementations
│   │   │   ├── base.py                  # MLModelWrapper interface
│   │   │   ├── xgboost_wrapper.py       # XGBoost
│   │   │   ├── nlinear_wrapper.py       # NLinear
│   │   │   ├── tft_wrapper.py           # Temporal Fusion Transformer
│   │   │   ├── catboost_wrapper.py      # CatBoost
│   │   │   └── model_factory.py         # ModelFactory
│   │   │
│   │   ├── pipeline/                    # Orchestration
│   │   │   ├── base.py                  # BasePipeline
│   │   │   ├── train.py                 # TrainPipeline
│   │   │   ├── eval.py                  # EvalPipeline
│   │   │   ├── test.py                  # TestPipeline (inference)
│   │   │   ├── engines/                 # Execution engines
│   │   │   │   ├── cv_runner.py         # Cross-validation
│   │   │   │   └── tuning_pipeline.py   # Optuna tuning
│   │   │   └── feature_ops/             # Feast operations
│   │   │       ├── ingest_batch_etth1.py   # Data ingestion
│   │   │       ├── materialize_etth1.py    # Offline→Online sync
│   │   │       └── ...
│   │   │
│   │   ├── preprocess/                  # Data Transformation
│   │   │   ├── transform_manager.py     # Stateful transformations
│   │   │   ├── transforms/              # Transform implementations
│   │   │   │   ├── fill_missing.py      # Imputation
│   │   │   │   ├── normalize.py         # Scaling
│   │   │   │   ├── label_encoding.py    # Categorical encoding
│   │   │   │   └── math_transforms.py   # Log, clip, etc.
│   │   │   └── offline.py               # Offline preprocessor
│   │   │
│   │   ├── tracking/                    # MLflow Integration
│   │   │   ├── mlflow_manager.py        # MLflow client wrapper
│   │   │   ├── pyfunc_wrapper.py        # PyFunc model wrapper
│   │   │   └── experiment_tracker.py    # Experiment logging
│   │   │
│   │   ├── trainer/                     # Training Loops
│   │   │   ├── base.py                  # BaseTrainer
│   │   │   ├── ml_trainer.py            # Sklearn-style models
│   │   │   ├── dl_trainer.py            # PyTorch models
│   │   │   └── trainer_factory.py       # TrainerFactory
│   │   │
│   │   ├── tuning/                      # Hyperparameter Tuning
│   │   │   ├── optuna_tuner.py          # Optuna integration
│   │   │   └── search_space.py          # Search space definitions
│   │   │
│   │   └── utils/                       # Helper Functions
│   │       ├── config_loader.py         # Hydra config loading
│   │       ├── func_utils.py            # General utilities
│   │       └── mlflow_utils.py          # MLflow helpers
│   │
│   └── run.py                           # CLI Entrypoint
│
├── tests/                               # Testing Suite
│   ├── conftest.py                      # Pytest fixtures
│   ├── test_datamodule.py               # Data loading tests
│   ├── test_models.py                   # Model tests
│   ├── test_pipeline.py                 # End-to-end tests
│   ├── test_preprocessing.py            # Transform tests
│   ├── test_feast_integration.py        # Feast tests
│   ├── test_ray_serve.py                # Ray Serve tests
│   └── test_api.py                      # API tests
│
├── docs/                                # Documentation
│   ├── architecture.md                  # System design
│   ├── offlineworkflow.md               # Training pipeline
│   ├── onlineworkflow.md                # Serving pipeline
│   ├── preprocessing.md                 # Transform guide
│   ├── adding_new_model.md              # Extension guide
│   └── directorystructure.md            # This file
│
├── Docker & Deployment
│   ├── Dockerfile                       # Multi-stage build
│   ├── docker-compose.yml               # Local development
│   └── .dockerignore                    # Build exclusions
│
├── Project Files
│   ├── requirements/
│   │   ├── prod.txt                     # Production dependencies
│   │   └── dev.txt                      # Development dependencies
│   ├── Makefile                         # Handy commands
│   ├── setup.py                         # Package installation
│   ├── pyproject.toml                   # Modern Python config
│   ├── .pre-commit-config.yaml          # Code quality hooks
│   ├── .gitignore                       # Git exclusions
│   └── README.md                        # Getting started
│
└── Generated Artifacts (Git-ignored)
    ├── artifacts/                       # Model artifacts
    │   └── models/
    ├── mlruns/                          # MLflow tracking
    └── .ray/                            # Ray runtime files
```
