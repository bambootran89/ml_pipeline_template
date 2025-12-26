"""
Centralized MLflow management module providing a unified, safe, high-level
interface for experiment tracking, run lifecycle, artifact logging, PyFunc
model persistence, and registry-based component loading.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union, cast

import mlflow
from omegaconf import DictConfig, OmegaConf

from mlproject.src.tracking.pyfunc_wrapper import ArtifactPyFuncWrapper


class MLflowManager:
    """
    Unified controller for all MLflow operations.

    When MLflow is disabled via configuration, all logging calls become silent
    no-ops while preserving program correctness.
    """

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]) -> None:
        """
        Initialize MLflow manager and configure the tracking backend.

        Args:
            cfg: Hydra DictConfig or a plain Python dictionary containing an
                 optional `mlflow` section.
        """
        self.cfg = cfg
        raw_cfg = cfg.get("mlflow", {}) if isinstance(cfg, dict) else cfg.mlflow
        self.mlflow_cfg = cast(Dict[str, Any], raw_cfg)
        self.enabled = bool(self.mlflow_cfg.get("enabled", False))

        if self.enabled:
            mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(
                self.mlflow_cfg.get("experiment_name", "default_experiment")
            )

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
    ) -> Generator[MLflowManager, None, None]:
        """
        Safely manage the MLflow run lifecycle.

        Args:
            run_name: Optional run name, falling back to config or `default_run`.
            nested: Allow nested MLflow runs.

        Yields:
            The same MLflowManager instance for unified logging.
        """
        if not self.enabled:
            yield self
            return

        name = run_name or self.mlflow_cfg.get("run_name", "default_run")

        if not nested and mlflow.active_run() is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass

        try:
            with mlflow.start_run(run_name=name, nested=nested):
                yield self
        except Exception as exc:
            try:
                mlflow.log_param("error", str(exc))
                mlflow.set_tag("status", "failed")
            except Exception:
                pass
            raise

    def log_metadata(
        self,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None,
    ) -> None:
        """
        Log parameters and metrics safely in a single unified call.

        Args:
            params: Key-value parameters (OmegaConf convertible allowed).
            metrics: Numeric metrics.
            step: Optional training step index.
        """
        if not self.enabled:
            return

        if params:
            converted = (
                OmegaConf.to_container(params, resolve=True)
                if isinstance(params, DictConfig) or hasattr(params, "__dict__")
                else params
            )
            clean_params = cast(Dict[str, Any], dict(converted))  # type: ignore
            mlflow.log_params(clean_params)

        if metrics:
            mlflow.log_metrics(metrics, step=step)

    def log_component(
        self,
        obj: Any,
        name: str,
        artifact_type: str = "model",
    ) -> None:
        """
        Persist a Python artifact into MLflow via PyFunc and register it.

        Args:
            obj: Python object exposing `predict()` or `transform()`.
            name: Artifact path and registry model name.
            artifact_type: 'model' -> use `predict`, otherwise use `transform`.
        """
        if not self.enabled:
            return

        method_name = "transform" if artifact_type == "preprocess" else "predict"
        method = getattr(obj, method_name, None)

        if not callable(method):
            raise AttributeError(
                f"Object is missing a callable '{method_name}' method for PyFunc."
            )

        wrapped = ArtifactPyFuncWrapper(obj, predict_method=method_name)

        try:
            mlflow.pyfunc.log_model(
                artifact_path=name,
                python_model=wrapped,
                registered_model_name=name,
            )
        except Exception as exc:
            print(f"[MLflowManager] Failed to log component '{name}': {exc}")

    def _fallback_latest(self, name: str, alias: str) -> Optional[Any]:
        """
        Fallback to loading the latest artifact from the most recent run.

        Args:
            name: Model name or artifact path.
            alias: Registry alias attempted before fallback.

        Returns:
            Loaded PyFunc model or None if fallback fails.
        """
        _ = alias
        try:
            runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
            if runs is None or (hasattr(runs, "empty") and runs.empty):  # type: ignore
                print(f"[MLflowManager] No runs found. Cannot fallback for '{name}'.")
                return None

            last_run = runs.iloc[0]  # type: ignore
            run_id = last_run.info.run_id  # <-- fixed access

            print(f"[MLflowManager] Latest run resolved: run_id='{run_id}'.")
            artifact_uri = f"runs:/{run_id}/{name}"

            print(
                f"[MLflowManager] Loading latest artifact for '{name}' "
                f"from '{artifact_uri}'."
            )

            loaded = mlflow.pyfunc.load_model(artifact_uri)
            print(f"[MLflowManager] Successfully loaded '{name}' from latest run.")
            return loaded

        except Exception as fb_exc:
            print(f"[MLflowManager] Latest fallback also failed for '{name}'.")
            print(f"[MLflowManager] Fallback error: {fb_exc}")
            return None

    def _unwrap(self, loaded: Any, name: str) -> Optional[Any]:
        """
        Unwrap the raw Python artifact from a PyFunc model if wrapped.

        Args:
            loaded: Loaded PyFunc model.
            name: Artifact/model name used during logging.

        Returns:
            The original unwrapped artifact or None if unwrap is not possible.
        """
        model_impl = getattr(loaded, "_model_impl", None)
        if model_impl is None:
            print(f"[MLflowManager] No _model_impl in '{name}'. Cannot unwrap.")
            return None

        python_model = getattr(model_impl, "python_model", None)
        print(
            f"[MLflowManager] PyFunc wrapper type detected: "
            f"'{type(python_model).__name__ if python_model else None}'."
        )

        if isinstance(python_model, ArtifactPyFuncWrapper):
            print(f"[MLflowManager] Unwrapping raw artifact from '{name}'.")
            return python_model.get_raw_artifact()

        print(f"[MLflowManager] '{name}' is not wrapped. Returning None.")
        return None

    def load_component(self, name: str, alias: str = "latest") -> Optional[Any]:
        """
        Load a PyFunc model using registry alias, else fallback to latest run.

        Args:
            name: Registry model name.
            alias: Registry alias (default: latest).

        Returns:
            The unwrapped raw artifact or None if loading or unwrapping fails.
        """
        if not self.enabled:
            print(f"[MLflowManager] MLflow disabled. Skipping load for '{name}'.")
            return None

        alias = alias.lower()
        model_uri = f"models:/{name}@{alias}"
        print(f"[MLflowManager] Attempting to load '{name}' using alias '{alias}'.")

        try:
            loaded = mlflow.pyfunc.load_model(model_uri)
            print(f"[MLflowManager] Loaded '{name}' via alias '{alias}'.")
        except Exception as exc:
            print(
                f"[MLflowManager] Alias '{alias}' not found for '{name}', "
                "falling back to latest."
            )
            print(f"[MLflowManager] Root error: {exc}")
            loaded = self._fallback_latest(name, alias)
            if loaded is None:
                return None

        if loaded is None:
            return None

        return self._unwrap(loaded, name)
