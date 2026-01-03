"""Hyperparameter tuning step using Optuna (BEST PRACTICE VERSION).

This implementation follows the clean separation pattern:
1. Tuning phase: Find best params in nested MLflow runs
2. Retrain phase: Train final model in separate run (handled by pipeline)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mlproject.src.datamodule.splitters.base import BaseSplitter
from mlproject.src.datamodule.splitters.timeseries import TimeSeriesFoldSplitter
from mlproject.src.pipeline.compat.v2.steps.base import BasePipelineStep
from mlproject.src.tracking.mlflow_manager import MLflowManager
from mlproject.src.tuning.optuna import OptunaTuner


class TunerStep(BasePipelineStep):
    """Run hyperparameter optimization using Optuna.

    This step performs cross-validation based hyperparameter search
    in a nested MLflow parent run, then stores best params for
    downstream model training.

    Key Design:
    -----------
    - Tuning trials are NESTED under parent run
    - Does NOT train final model (separation of concerns)
    - Best params stored in context for downstream steps
    - Final model training handled by separate ModelTrainingStep

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Preprocessed data (optional - tuner loads its own data).

    Context Outputs
    ---------------
    <step_id>_best_params : Dict[str, Any]
        Best hyperparameters found.
    <step_id>_best_value : float
        Best metric value achieved.
    <step_id>_study : optuna.Study
        Optuna study object with all trials.

    Configuration Parameters
    ------------------------
    n_trials : int, optional
        Number of trials to run (default: from tuning config).

    Examples
    --------
    YAML configuration::

        pipeline:
          steps:
            # Stage 1: Find best params
            - id: "tune_model"
              type: "tuning"
              enabled: true
              depends_on: ["preprocess"]

            # Stage 2: Train with best params
            - id: "train_best"
              type: "model"
              enabled: true
              depends_on: ["tune_model"]
              use_tuned_params: true  # Read from tune_model_best_params

            # Stage 3: Evaluate
            - id: "evaluate"
              type: "evaluator"
              enabled: true
              model_step_id: "train_best"
    """

    def __init__(self, *args, n_trials: Optional[int] = None, **kwargs) -> None:
        """Initialize tuning step.

        Parameters
        ----------
        n_trials : int, optional
            Override number of trials from config.
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.n_trials_override = n_trials

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        This follows the clean pattern:
        1. Start parent MLflow run
        2. Run Optuna tuning (trials nested as children)
        3. Log best params to parent run
        4. Store best params in context
        5. Let downstream ModelTrainingStep retrain with best params

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with tuning results added.
        """
        self.validate_dependencies(context)

        print(f"\n{'=' * 60}")
        print(f"[{self.step_id}] HYPERPARAMETER TUNING")
        print(f"{'=' * 60}\n")

        # Build CV splitter
        data_type = self.cfg.get("data", {}).get("type", "timeseries")
        n_splits = self.cfg.get("tuning", {}).get("n_splits", 3)

        splitter: BaseSplitter
        if data_type == "timeseries":
            splitter = TimeSeriesFoldSplitter(self.cfg, n_splits=n_splits)
        else:
            splitter = BaseSplitter(self.cfg, n_splits=n_splits)

        # Initialize MLflow manager
        mlflow_manager = MLflowManager(self.cfg)

        # Get optimization settings
        metric_name = self.cfg.get("tuning", {}).get("optimize_metric", "mae_mean")
        direction = self.cfg.get("tuning", {}).get("direction", "minimize")
        n_trials = self.n_trials_override or self.cfg.get("tuning", {}).get(
            "n_trials", 50
        )
        timeout = self.cfg.get("tuning", {}).get("timeout")

        print(f"[{self.step_id}] Configuration:")
        print(f"  - Trials: {n_trials}")
        print(f"  - Metric: {metric_name} ({direction})")
        print(f"  - CV folds: {n_splits}")

        # ========================================
        # START PARENT RUN (best practice pattern)
        # ========================================
        experiment_name = self.cfg.experiment.get("name", "undefined")
        with mlflow_manager.start_run(run_name=f"Hparam_Tuning_{experiment_name}"):
            print(f"\n[MLflow] Started parent run: Hparam_Tuning_{experiment_name}")

            # Create tuner (trials will be nested children)
            tuner = OptunaTuner(
                cfg=self.cfg,
                splitter=splitter,
                mlflow_manager=mlflow_manager,
                metric_name=metric_name,
                direction=direction,
            )

            # Run tuning
            print(f"\n[{self.step_id}] Running optimization...")
            result = tuner.tune(n_trials=n_trials, timeout=timeout)

            # Log best params to parent run for quick view
            mlflow_manager.log_metadata(params=result["best_params"])

            print("\n[MLflow] Logged best params to parent run")

        # ========================================
        # Store results in context
        # ========================================
        context[f"{self.step_id}_best_params"] = result["best_params"]
        context[f"{self.step_id}_best_value"] = result["best_value"]
        context[f"{self.step_id}_study"] = result["study"]

        print(f"\n{'=' * 60}")
        print(f"[{self.step_id}] TUNING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Best {metric_name}: {result['best_value']:.6f}")
        print("\n  Best parameters:")
        for param, value in result["best_params"].items():
            print(f"    - {param}: {value}")
        print(f"\n{'=' * 60}\n")

        return context
