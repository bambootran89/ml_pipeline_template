"""
DataModule pipeline step for constructing a reusable DataModule instance,
optionally generating model-based features, and wiring results back into
the shared pipeline execution context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class DataModuleStep(BasePipelineStep):
    """
    Build a DataModule from the pipeline context and optionally generate
    features using a fitted model.

    This step supports:
    1. Restoring a previously fitted DataModule from MLflow via `instance_key`.
    2. Constructing a new DataModule when no restored instance is available.
    3. Generating model predictions and exposing them as engineered features
       for downstream evaluation or inference steps.

    Expected Context Inputs
    ----------------------
    data : pd.DataFrame or array-like, optional
        Input dataset for building the DataModule. If not explicitly wired,
        the step falls back to the key `preprocessed_data` from context.
    model : Any, optional
        A fitted model instance capable of producing predictions via `predict()`.

    Context Outputs
    ---------------
    datamodule : Any
        The constructed or restored DataModule instance.
    features : np.ndarray, optional
        Engineered features generated from model predictions if
        `output_as_feature` is enabled.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        output_as_feature: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DataModuleStep.

        Parameters
        ----------
        step_id : str
            Unique pipeline step identifier.
        cfg : DictConfig or compatible config object
            Configuration containing data and evaluation settings.
        enabled : bool, default=True
            Whether this step should execute in the pipeline.
        depends_on : Optional[List[str]], default=None
            List of prerequisite step IDs that must complete before execution.
        output_as_feature : bool, default=False
            Whether model predictions should be stored as engineered features.
        **kwargs : Any
            Additional arguments passed to the BasePipelineStep.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.output_as_feature = output_as_feature

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the DataModule construction and optional feature generation.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline execution context.

        Returns
        -------
        Dict[str, Any]
            Updated pipeline context containing DataModule and optional features.

        Raises
        ------
        ValueError
            If required data is missing from context.
        """
        self.validate_dependencies(context)

        # Resolve input dataset
        df_features: pd.DataFrame = self.get_input(context, "features")
        df_targets: pd.DataFrame = self.get_input(context, "targets")
        data_cfg: Dict[str, Any] = self.cfg.get("data", {})
        data_type: str = str(data_cfg.get("type", "tabular")).lower()

        if data_type == "timeseries":
            input_df = df_features.copy()
        else:
            input_df = pd.concat([df_features, df_targets], axis=1)
        df = self._resolve_dataframe(context, input_df)

        # Build DataModule
        datamodule = self._build_datamodule(context, df)

        # Optional feature engineering from model predictions
        if self.output_as_feature:
            model = context.get("model")
            if model is not None:
                features = self._generate_features(datamodule, model)
                context[f"{self.step_id}_features"] = features

        return context

    def _resolve_dataframe(
        self,
        context: Dict[str, Any],
        input_df: Optional[Any],
    ) -> pd.DataFrame:
        """
        Normalize the input dataset into a pandas DataFrame.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline execution context.
        input_df : Optional[Any]
            Wired dataframe or array-like input.

        Returns
        -------
        pd.DataFrame
            Normalized dataframe.

        Raises
        ------
        ValueError
            If no valid input data is found in context.
        """
        if isinstance(input_df, pd.DataFrame):
            return input_df

        fallback_df = context.get("preprocessed_data")
        if isinstance(fallback_df, pd.DataFrame):
            return fallback_df

        if fallback_df is not None:
            return pd.DataFrame(fallback_df)

        raise ValueError(f"Step '{self.step_id}': no input data found for DataModule.")

    def _build_datamodule(
        self,
        context: Dict[str, Any],
        df: pd.DataFrame,
    ) -> Any:
        """
        Build and register a DataModule instance.

        Parameters
        ----------
        context : Dict[str, Any]
            Shared pipeline execution context.
        df : pd.DataFrame
            Input dataframe for DataModule construction.

        Returns
        -------
        Any
            Constructed DataModule instance.
        """
        print(f"[{self.step_id}] Building DataModule")
        dm = DataModuleFactory.build(self.cfg, df)
        dm.setup()
        self.set_output(context, "datamodule", dm)
        return dm

    def _generate_features(
        self,
        datamodule: Any,
        model: Any,
    ) -> np.ndarray:
        """
        Generate model predictions and expose them as numpy features.

        Parameters
        ----------
        datamodule : Any
            DataModule instance containing test data.
        model : Any
            Fitted model capable of producing predictions.

        Returns
        -------
        np.ndarray
            Prediction outputs formatted as feature vectors.
        """
        if hasattr(datamodule, "get_test_windows"):
            x_data, _ = datamodule.get_test_windows()
        else:
            x_data = datamodule.get_data()[-2]

        preds = model.predict(x_data)
        return np.asarray(preds, dtype=float)


StepFactory.register("datamodule", DataModuleStep)
