"""
Run lifecycle and environment metadata logging for MLflow.

Provides RunManager to manage MLflow runs, including context-managed
lifecycle and logging of reproducibility metadata.
"""

import contextlib
import getpass
from typing import Generator, Optional

import mlflow
import pandas as pd

# Optional import of git for logging git metadata
try:
    import git

    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False


class RunManager:
    """
    Manages MLflow run lifecycle and environment metadata logging.

    Attributes:
        mlflow_cfg (dict): MLflow-specific configuration block.
        enabled (bool): Flag indicating whether MLflow tracking is enabled.
    """

    def __init__(self, mlflow_cfg: dict, enabled: bool):
        """
        Initialize RunManager.

        Args:
            mlflow_cfg (dict): MLflow configuration dictionary.
            enabled (bool): Whether MLflow tracking is enabled.
        """
        self.mlflow_cfg = mlflow_cfg
        self.enabled = enabled

    @contextlib.contextmanager
    def start_run(
        self, run_name: Optional[str] = None
    ) -> Generator[Optional[mlflow.ActiveRun], None, None]:
        """
        Start an MLflow run within a context manager.

        Automatically generates a timestamped run name if none is provided,
        and logs environment metadata for reproducibility.

        Args:
            run_name (Optional[str]):
                Display name for the run. Defaults to a timestamped name.

        Yields:
            Optional[mlflow.ActiveRun]: Active MLflow run if enabled, else None.
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

    def _log_environment_info(self) -> None:
        """
        Log environment metadata for reproducibility.

        Includes:
            - Git commit hash and dirty flag (if git is available)
            - Username of the executing environment
        """
        if _GIT_AVAILABLE:
            try:
                repo = git.Repo(search_parent_directories=True)
                sha = repo.head.object.hexsha
                mlflow.set_tag("git_commit", sha)

                if repo.is_dirty():
                    mlflow.set_tag("git_dirty", "True")
            except Exception:
                pass

        mlflow.set_tag("user", getpass.getuser())
