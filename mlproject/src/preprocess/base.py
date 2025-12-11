"""
Base preprocessing cho offline training và online serving.

Fixed:
- Deprecated fillna(method=...) → ffill() + bfill()
- Feature name warning với StandardScaler
- Added: Logic download scaler từ MLflow nếu không tìm thấy local.
"""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ARTIFACT_DIR = os.path.join("mlproject", "artifacts", "preprocessing")


class PreprocessBase:
    """
    Base preprocessing logic used for BOTH offline training and online serving.
    """

    def __init__(self, cfg=None):
        """
        Initialize the preprocessing base object.
        """
        self.cfg = cfg or {}

        self.steps = self.cfg.get("preprocessing", {}).get("steps", [])

        self.artifacts_dir = self.cfg.get("preprocessing", {}).get(
            "artifacts_dir", ARTIFACT_DIR
        )
        self.scaler = None
        self.scaler_columns = None

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessing steps on the DataFrame."""
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        df = self._apply_fit_scaler(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to the DataFrame."""
        df = self._apply_fill_missing(df)
        df = self._apply_generate_covariates(df)
        if self.scaler is None:
            self.load_scaler()
        df = self._apply_scaling(df)
        return df

    def _apply_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation according to config."""
        step = self._get_step("fill_missing")
        if not step:
            return df

        method = step.get("method", "ffill")

        if method == "ffill":
            return df.ffill().bfill()

        if method == "mean":
            return df.fillna(df.mean())

        raise ValueError(f"Unknown fill_missing method: {method}")

    def _apply_generate_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional covariates based on config."""
        step = self._get_step("gen_covariates")
        if not step:
            return df

        cov = step.get("covariates", {})

        if "future" in cov and "day_of_week" in cov["future"]:
            if isinstance(df.index, pd.DatetimeIndex):
                df["day_of_week"] = df.index.dayofweek
        return df

    def _apply_fit_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and save to artifacts."""
        step = self._get_step("normalize")
        if not step:
            return df

        cols = self._get_numeric_columns(df, step)
        method = step.get("method", "zscore")

        if method == "zscore":
            scaler = StandardScaler()
            scaler.fit(df[cols].values)
            try:
                scaler.feature_names_in_ = np.array(cols)
            except Exception:
                pass
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaler.fit(df[cols].values)
            try:
                scaler.feature_names_in_ = np.array(cols)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown normalize method: {method}")

        self.scaler = scaler
        self.scaler_columns = cols

        self.save_scaler()

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted scaler."""
        if self.scaler is None:
            return df

        # Ensure all expected columns exist
        for c in self.scaler_columns:
            if c not in df.columns:
                df[c] = 0.0

        df[self.scaler_columns] = self.scaler.transform(df[self.scaler_columns])
        return df

    def save_scaler(self):
        """Save scaler + column list to artifact directory."""
        os.makedirs(self.artifacts_dir, exist_ok=True)
        with open(os.path.join(self.artifacts_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(
                {"scaler": self.scaler, "columns": self.scaler_columns},
                f,
            )

    def load_scaler(self):
        """
        Load scaler from artifact directory.
        If local file is missing, attempts to download from MLflow.
        """
        path = os.path.join(self.artifacts_dir, "scaler.pkl")

        # 1. Nếu không thấy local, thử download từ MLflow
        if not os.path.exists(path):
            print(
                f"[PreprocessBase] Local scaler not found at {path}. Checking MLflow..."
            )
            self._download_scaler_from_mlflow()

        # 2. Load nếu file tồn tại
        if not os.path.exists(path):
            print(
                "[PreprocessBase] Scaler artifact not found (Local or MLflow).\
                      Skipping scaling."
            )
            return

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            self.scaler = obj["scaler"]
            self.scaler_columns = obj["columns"]
            print(f"[PreprocessBase] Scaler loaded successfully from {path}")
        except Exception as e:
            print(f"[PreprocessBase] Error loading pickle file: {e}")

    def _download_scaler_from_mlflow(self):
        """
        Attempt to download scaler artifact from MLflow (Run or Registry).
        """
        mlflow_cfg = self.cfg.get("mlflow", {})
        if not mlflow_cfg.get("enabled", False):
            return

        try:
            client = MlflowClient()
            run_id = mlflow_cfg.get("run_id")

            # Nếu không có run_id, thử tìm từ Model Registry
            if not run_id:
                registry_conf = mlflow_cfg.get("registry", {})
                model_name = registry_conf.get("model_name")
                if model_name:
                    print(
                        f"""[PreprocessBase] Looking up latest
                        run for model '{model_name}'..."""
                    )
                    versions = client.get_latest_versions(
                        model_name, stages=["None", "Staging", "Production"]
                    )
                    if versions:
                        # Lấy version mới nhất (thường là cuối list)
                        latest_version = sorted(versions, key=lambda x: int(x.version))[
                            -1
                        ]
                        run_id = latest_version.run_id
                        print(
                            f"""[PreprocessBase] Found latest version {
                                latest_version.version} (Run ID: {run_id})"""
                        )

            if not run_id:
                print(
                    "[PreprocessBase] No run_id or registered model found. \
                        Cannot download scaler."
                )
                return

            # Artifact path trên MLflow: preprocessing/scaler/scaler.pkl
            artifact_rel_path = "preprocessing/scaler/scaler.pkl"

            print(
                f"""[PreprocessBase] Downloading \
                  artifact '{artifact_rel_path}' from run {run_id}..."""
            )

            # Download vào thư mục tạm, sau đó copy về artifacts_dir
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = client.download_artifacts(
                    run_id=run_id, path=artifact_rel_path, dst_path=tmp_dir
                )

                os.makedirs(self.artifacts_dir, exist_ok=True)
                final_path = os.path.join(self.artifacts_dir, "scaler.pkl")
                if os.path.exists(local_path):
                    shutil.copy2(local_path, final_path)
                    print(f"[PreprocessBase] Scaler downloaded to {final_path}")
                else:
                    print(
                        f"""[PreprocessBase] Artifact downloaded
                        but file not found at {local_path}"""
                    )

        except Exception as e:
            print(f"[PreprocessBase] Error downloading from MLflow: {e}")

    def _get_step(self, name: str):
        """Retrieve a preprocessing step by name."""
        return next((s for s in self.steps if s.get("name") == name), None)

    def _get_numeric_columns(self, df, step):
        """Get numeric columns to scale."""
        cols = step.get("columns")
        if cols:
            return cols

        return df.select_dtypes(include=[np.number]).columns.tolist()
