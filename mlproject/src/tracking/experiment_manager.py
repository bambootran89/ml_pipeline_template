"""
Experiment management for MLflow.

Provides ExperimentManager to handle creation and restoration
of MLflow experiments in a safe and automated way.
"""

from typing import Any, Optional

import mlflow


class ExperimentManager:
    """
    Handles creating or restoring MLflow experiments.

    Features:
        - Create new experiments if they do not exist.
        - Restore deleted experiments automatically.
        - Fallback to a new versioned experiment name if restoration fails.
    """

    def __init__(self, mlflow_cfg: Any, enabled: bool):
        """
        Initialize ExperimentManager.

        Args:
            mlflow_cfg (Any): Configuration dictionary or OmegaConf object
                containing at least the 'experiment_name' field.
            enabled (bool): Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    def setup_experiment(self) -> Optional[mlflow.entities.Experiment]:
        """
        Create or restore an MLflow experiment.

        The method tries to set the experiment with the given name.
        If the experiment was deleted, it attempts to restore it.
        If restoration fails, it creates a new experiment with a
        "_v2" suffix.

        Returns:
            mlflow.entities.Experiment | None:
                The MLflow Experiment object if enabled, else None.
        """
        if not self.enabled:
            return None

        exp_name = self.mlflow_cfg.get("experiment_name", "Default_Experiment")

        try:
            experiment = mlflow.set_experiment(exp_name)
            return experiment

        except mlflow.exceptions.MlflowException as e:
            if "deleted experiment" not in str(e).lower():
                raise e

            print(f"[MLflow] Experiment '{exp_name}' deleted → restoring…")

            client = mlflow.MlflowClient()
            deleted = client.search_experiments(
                view_type=mlflow.entities.ViewType.DELETED_ONLY
            )

            for exp in deleted:
                if exp.name == exp_name:
                    client.restore_experiment(exp.experiment_id)
                    return mlflow.set_experiment(exp_name)

            # fallback to new versioned experiment
            new_name = f"{exp_name}_v2"
            print(f"[MLflow] Could not restore → creating {new_name}")
            return mlflow.set_experiment(new_name)
