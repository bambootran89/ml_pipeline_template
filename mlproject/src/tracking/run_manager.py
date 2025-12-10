"""
Run lifecycle & environment metadata logging for MLflow.
"""

import contextlib
import getpass

import mlflow
import pandas as pd


class RunManager:
    """
    Provides:
    - MLflow start_run() context manager
    - Environment info logging (git commit, user)
    """

    def __init__(self, mlflow_cfg):
        """
        Args:
            mlflow_cfg: MLflow config block.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = mlflow_cfg.get("enabled", False)

    @contextlib.contextmanager
    def start_run(self, run_name=None):
        """
        Context manager wrapper around mlflow.start_run().

        Args:
            run_name: Optional run display name.

        Yields:
            Active MLflow run.
        """
        if not self.enabled:
            yield None
            return

        if run_name is None:
            prefix = self.mlflow_cfg.get("run_name_prefix", "run")
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{prefix}_{timestamp}"

        with mlflow.start_run(run_name=run_name) as run:
            self._log_environment_info()
            yield run

    def _log_environment_info(self):
        """
        Log metadata for reproducibility:
        - Git commit & dirty flag
        - Username
        """
        try:
            import git

            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            mlflow.set_tag("git_commit", sha)

            if repo.is_dirty():
                mlflow.set_tag("git_dirty", "True")
        except Exception:
            pass

        mlflow.set_tag("user", getpass.getuser())
