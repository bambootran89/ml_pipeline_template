"""Pipeline profiling step for output analysis.

This module provides comprehensive profiling of pipeline outputs including:
- Cluster distribution analysis
- Prediction statistics
- Model performance summaries
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class ProfilingStep(BasePipelineStep):
    """Profile and summarize pipeline outputs.

    Context Inputs
    --------------
    All keys from previous steps are analyzed automatically.

    Context Outputs
    ---------------
    <step_id>_profile : Dict[str, Any]
        Profiling report with statistics.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        include_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize profiling step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration object.
        enabled : bool
            Whether step is active.
        depends_on : Optional[List[str]]
            Prerequisite steps.
        include_keys : Optional[List[str]]
            Specific context keys to profile.
        exclude_keys : Optional[List[str]]
            Context keys to exclude from profiling.
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.include_keys = include_keys or []
        self.exclude_keys = exclude_keys or ["cfg", "preprocessor"]

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Profile pipeline outputs and generate report.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context with all step outputs.

        Returns
        -------
        Dict[str, Any]
            Context with profiling report added.
        """
        self.validate_dependencies(context)

        profile: Dict[str, Any] = {
            "summary": {},
            "clusters": {},
            "predictions": {},
            "metrics": {},
            "data_quality": {},
        }

        keys_to_profile = self._get_keys_to_profile(context)

        for key in keys_to_profile:
            self._profile_value(key, context[key], profile)

        self._print_profile(profile)
        self.set_output(context, "profile", profile)

        return context

    def _get_keys_to_profile(self, context: Dict[str, Any]) -> List[str]:
        """Determine which context keys to profile.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        List[str]
            Keys to include in profiling.
        """
        if self.include_keys:
            return [k for k in self.include_keys if k in context]
        return [k for k in context if k not in self.exclude_keys]

    def _profile_value(
        self,
        key: str,
        value: Any,
        profile: Dict[str, Any],
    ) -> None:
        """Profile a single context value.

        Parameters
        ----------
        key : str
            Context key name.
        value : Any
            Value to profile.
        profile : Dict[str, Any]
            Profile dict to update.
        """
        if key.endswith("_metrics"):
            profile["metrics"][key] = self._summarize_metrics(value)
        elif "prediction" in key.lower() or "pred" in key.lower():
            profile["predictions"][key] = self._analyze_predictions(value)
        elif isinstance(value, pd.DataFrame):
            profile["data_quality"][key] = self._analyze_dataframe(value)
        elif isinstance(value, np.ndarray):
            # Only analyze as clusters if 1D integer-like array
            if value.ndim == 1 and np.issubdtype(value.dtype, np.integer):
                if "cluster" in key.lower() or "label" in key.lower():
                    profile["clusters"][key] = self._analyze_clusters(value)
                else:
                    profile["summary"][key] = self._analyze_array(value)
            else:
                profile["summary"][key] = self._analyze_array(value)

    def _summarize_metrics(self, metrics: Any) -> Dict[str, Any]:
        """Summarize evaluation metrics.

        Parameters
        ----------
        metrics : Any
            Raw metrics dict.

        Returns
        -------
        Dict[str, Any]
            Summarized metrics with rankings.
        """
        if not isinstance(metrics, dict):
            return {"raw": str(metrics)}

        summary: Dict[str, Any] = {"values": {}, "best": None, "worst": None}
        numeric_metrics: Dict[str, float] = {}

        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                summary["values"][k] = round(float(v), 6)
                numeric_metrics[k] = float(v)

        if numeric_metrics:
            sorted_keys = sorted(
                numeric_metrics.keys(), key=lambda x: numeric_metrics[x]
            )
            summary["best"] = sorted_keys[0]
            summary["worst"] = sorted_keys[-1]

        return summary

    def _analyze_clusters(self, labels: Any) -> Dict[str, Any]:
        """Analyze cluster label distribution.

        Parameters
        ----------
        labels : Any
            Cluster labels array.

        Returns
        -------
        Dict[str, Any]
            Cluster distribution statistics.
        """
        if labels is None:
            return {"error": "No labels provided"}

        # Skip non-array types
        if isinstance(labels, dict):
            return {"error": "Expected array, got dict"}
        if not hasattr(labels, "__iter__") or isinstance(labels, str):
            return {"error": f"Expected array, got {type(labels).__name__}"}

        try:
            arr = np.asarray(labels).flatten()
        except (ValueError, TypeError) as exc:
            return {"error": f"Cannot convert to array: {exc}"}
        unique, counts = np.unique(arr, return_counts=True)
        distribution = {int(u): int(c) for u, c in zip(unique, counts)}
        total = len(arr)
        percentages = {k: round(v / total * 100, 2) for k, v in distribution.items()}

        return {
            "n_clusters": len(unique),
            "n_samples": total,
            "distribution": distribution,
            "percentages": percentages,
            "min_cluster_size": int(counts.min()),
            "max_cluster_size": int(counts.max()),
            "std_cluster_size": round(float(counts.std()), 2),
            "balance_ratio": round(float(counts.min() / counts.max()), 4),
        }

    def _analyze_predictions(self, preds: Any) -> Dict[str, Any]:
        """Analyze prediction statistics.

        Parameters
        ----------
        preds : Any
            Prediction array.

        Returns
        -------
        Dict[str, Any]
            Prediction statistics.
        """
        if preds is None:
            return {"error": "No predictions provided"}

        arr = np.asarray(preds).flatten()

        return {
            "n_samples": len(arr),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "q25": round(float(np.percentile(arr, 25)), 6),
            "q75": round(float(np.percentile(arr, 75)), 6),
        }

    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame quality.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to analyze.

        Returns
        -------
        Dict[str, Any]
            Data quality metrics.
        """
        return {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_per_column": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        }

    def _analyze_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Analyze numpy array.

        Parameters
        ----------
        arr : np.ndarray
            Array to analyze.

        Returns
        -------
        Dict[str, Any]
            Array statistics.
        """
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": int(arr.size),
        }

    def _print_profile(self, profile: Dict[str, Any]) -> None:
        """Print profiling report to console.

        Parameters
        ----------
        profile : Dict[str, Any]
            Profiling results.
        """
        print(f"\n{'=' * 60}")
        print(f"[{self.step_id}] PIPELINE PROFILING REPORT")
        print(f"{'=' * 60}")

        if profile["metrics"]:
            print("\n[Metrics Summary]")
            for key, data in profile["metrics"].items():
                print(f"  {key}:")
                for metric, value in data.get("values", {}).items():
                    print(f"    - {metric}: {value}")

        if profile["clusters"]:
            print("\n[Cluster Analysis]")
            for key, data in profile["clusters"].items():
                print(f"  {key}:")
                print(f"    - n_clusters: {data.get('n_clusters')}")
                print(f"    - balance_ratio: {data.get('balance_ratio')}")
                print(f"    - distribution: {data.get('distribution')}")

        if profile["predictions"]:
            print("\n[Prediction Statistics]")
            for key, data in profile["predictions"].items():
                print(f"  {key}:")
                print(f"    - mean: {data.get('mean')}, std: {data.get('std')}")
                print(f"    - range: [{data.get('min')}, {data.get('max')}]")

        print(f"\n{'=' * 60}\n")


StepFactory.register("profiling", ProfilingStep)
