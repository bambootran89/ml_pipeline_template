import os
import pickle
import shutil
import tempfile
from typing import Any, List, Optional, Tuple

from mlflow.tracking import MlflowClient


class ScalerManager:
    """Manage persistence of preprocessing scalers.

    This class handles:
    - Saving fitted scalers to local artifacts.
    - Loading scalers from local storage or MLflow.
    - Downloading scaler artifacts from MLflow runs.

    Purpose:
        Decouple scaler lifecycle management from preprocessing logic.
    """

    def __init__(self, artifacts_dir: str, cfg: dict):
        """Initialize the scaler manager.

        Args:
            artifacts_dir (str): Directory where scaler artifacts are stored.
            cfg (dict): Configuration containing MLflow settings.
        """
        self.artifacts_dir = artifacts_dir
        self.cfg = cfg
        self.scaler: Optional[Any] = None
        self.scaler_columns: Optional[List[str]] = None

    def save(self, scaler: Any, columns: List[str]) -> None:
        """Save a fitted scaler and associated column names.

        Args:
            scaler (Any): Fitted sklearn scaler instance.
            columns (List[str]): List of feature names used during fitting.
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        path = os.path.join(self.artifacts_dir, "scaler.pkl")

        with open(path, "wb") as f:
            pickle.dump({"scaler": scaler, "columns": columns}, f)

        self.scaler = scaler
        self.scaler_columns = columns

    def load(self) -> Tuple[Optional[Any], Optional[List[str]]]:
        """Load a scaler from local artifacts or MLflow.

        Logic:
        1. Attempt to load from local filesystem.
        2. If missing, attempt to download from MLflow.
        3. Return loaded scaler and column list, if present.

        Returns:
            Tuple[Optional[Any], Optional[List[str]]]:
                - scaler: Loaded sklearn scaler, or None
                - columns: List of feature names, or None
        """
        path = os.path.join(self.artifacts_dir, "scaler.pkl")

        if not os.path.exists(path):
            print("[ScalerManager] Local scaler not found. Checking MLflow...")
            self._download_from_mlflow()

        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.scaler = obj["scaler"]
            self.scaler_columns = obj["columns"]
            print(f"[ScalerManager] Loaded from {path}")
            return self.scaler, self.scaler_columns

        print("[ScalerManager] No scaler found (local or MLflow).")
        return None, None

    def _download_from_mlflow(self) -> None:
        """Download the scaler artifact from an MLflow run.

        Behavior:
        - Uses MLflow run ID if specified.
        - Otherwise, attempts to fetch the latest registered model version.
        - Downloads `preprocessing/scaler/scaler.pkl` into the artifacts directory.
        """
        mlflow_cfg = self.cfg.get("mlflow", {})
        if not mlflow_cfg.get("enabled", False):
            return

        try:
            client = MlflowClient()
            run_id = mlflow_cfg.get("run_id")

            # Attempt to fetch run_id from model registry
            if not run_id:
                registry_conf = mlflow_cfg.get("registry", {})
                model_name = registry_conf.get("model_name")
                if model_name:
                    versions = client.get_latest_versions(
                        model_name, stages=["None", "Staging", "Production"]
                    )
                    if versions:
                        latest = sorted(versions, key=lambda x: int(x.version))[-1]
                        run_id = latest.run_id

            if not run_id:
                return

            artifact_path = "preprocessing/scaler/scaler.pkl"
            print(f"[ScalerManager] Downloading from MLflow run {run_id}...")

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = client.download_artifacts(
                    run_id=run_id,
                    path=artifact_path,
                    dst_path=tmp_dir,
                )

                os.makedirs(self.artifacts_dir, exist_ok=True)
                final_path = os.path.join(self.artifacts_dir, "scaler.pkl")

                if os.path.exists(local_path):
                    shutil.copy2(local_path, final_path)
                    print(f"[ScalerManager] Downloaded to {final_path}")

        except Exception as e:
            print(f"[ScalerManager] MLflow download failed: {e}")
