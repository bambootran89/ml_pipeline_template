# Architecture & Design Philosophy (Core Design Principles)

## Unified Artifact Packaging (Training–Serving Skew Prevention)

This system enforces **strict versioned coupling between preprocessing and modeling** by packaging them together as a **single MLflow artifact**.

* **Self-Contained, Versioned Artifacts**
  Instead of logging only model weights, each MLflow run produces a **fully self-contained artifact** that includes:

  * The trained **Stateful Preprocessor** (with fitted statistics)
  * The trained **Model**

  Both components are wrapped and versioned together inside a single **PyFuncWrapper**, ensuring they are **always deployed as a unit**.

* **Explicit Training–Serving Contract**
  Every artifact version defines an immutable contract:

  > *Given raw input data → apply the exact preprocessing learned at training time → run model inference.*

  This eliminates ambiguity about which preprocessing logic or parameters were used for a given model version.

* **Zero Logic Duplication Across Environments**
  The serving environment does **not reimplement or reinterpret** any data processing logic. It simply **loads (hydrates) the artifact produced during training** via MLflow.

  As a result:

  * No drift caused by mismatched preprocessing code
  * No hidden differences between offline experiments and online inference

* **Training–Serving Skew Prevention by Design**
  By forcing preprocessing and modeling to be **co-versioned, co-deployed, and co-loaded**, the system guarantees **behavioral consistency** between:

  * Training
  * Validation
  * Batch inference
  * Online serving

This design treats **preprocessing + model = one inseparable versioned unit**, making MLflow the **single source of truth** for both data transformation and inference logic.


## Distributed & Async Architecture

- **Decoupled Microservices**:
  Using **Ray Serve**, Feature Engineering (CPU-bound) and Model Inference (GPU-bound) are split into independent **Actors**, allowing each to scale independently based on load.

- **Non-blocking I/O**:
  Heavy computation is handled with **Async/Await** and **ThreadPool**, preventing the API's main event loop from being blocked while executing CPU-intensive tasks.

## Modular & Configuration-Driven

- **Hydra Configs**:
  All experiment parameters, model architectures, and dataset paths are managed through **YAML** configuration files, enabling reproducibility and flexible experiment management.

- **Factory Pattern**:
  The training pipeline leverages the **Factory Pattern** (e.g., `ModelFactory`, `TrainerFactory`) to separate **instantiation** from execution logic. This design makes it easy to add new algorithms without modifying core pipeline code.


# System Workflow

## Training & Tuning Pipeline (Offline Workflow)
The training pipeline is orchestrated by Hydra and powered by a robust factory pattern. It supports Nested Cross-Validation, Hyperparameter Tuning (Optuna), and automatic artifact logging to MLflow.

[Offline Workflow](docs/offlineworkflow.md)

- Factory Pattern: Seamlessly switch between XGBoost, TFT, or NLinear via config model: name.
## Inference & Serving Pipeline (Online Workflow)
The serving layer is designed to be stateless and reproducible. It strictly uses the artifacts (Models & Scalers) generated during the training phase to ensure the Training-Serving Skew is minimized.

[Online Workflow](docs/onlineworkflow.md)


## Key Features
- Engineered for Reliability: Includes Type Hinting (mypy), Linting (flake8, pylint), and Unit Tests (pytest).

- Experiment Tracking: Built-in integration with MLflow for logging metrics, params, and artifacts.

- Scalable Serving: Supports both lightweight FastAPI and distributed Ray Serve.
