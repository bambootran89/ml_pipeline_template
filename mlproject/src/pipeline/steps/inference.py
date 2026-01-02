"""Inference step for test/prediction pipelines.

This module provides a step that generates predictions without
evaluation metrics, useful for production inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mlproject.src.pipeline.steps.base import BasePipelineStep


class InferenceStep(BasePipelineStep):
    """Generate predictions without computing evaluation metrics.

    This step runs inference using a loaded model and optionally
    saves predictions to disk.

    Context Inputs
    --------------
    preprocessed_data : pd.DataFrame
        Preprocessed features.
    <model_step_id>_model : ModelWrapper
        Loaded model instance.

    Context Outputs
    ---------------
    <step_id>_predictions : np.ndarray
        Raw model predictions.

    Configuration Parameters
    ------------------------
    model_step_id : str, required
        ID of step that loaded the model.
    save_path : str, optional
        Path to save predictions CSV file.
    include_inputs : bool, default=False
        If True, save input features alongside predictions.

    Examples
    --------
    YAML configuration::

        steps:
          - id: "inference"
            type: "inference"
            enabled: true
            depends_on: ["load_model", "preprocess"]
            model_step_id: "load_model"
            save_path: "outputs/predictions.csv"
            include_inputs: false
    """

    def __init__(
        self,
        *args,
        model_step_id: Optional[str] = None,
        save_path: Optional[str] = None,
        include_inputs: bool = False,
        **kwargs,
    ) -> None:
        """Initialize inference step.

        Parameters
        ----------
        model_step_id : str, optional
            ID of step that loaded the model. Required for execution.
        save_path : str, optional
            Path to save predictions CSV.
        include_inputs : bool, default=False
            Whether to save input features with predictions.
        *args, **kwargs
            Passed to BasePipelineStep.
        """
        super().__init__(*args, **kwargs)
        self.model_step_id = model_step_id
        self.save_path = save_path
        self.include_inputs = include_inputs

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference and optionally save predictions.

        Parameters
        ----------
        context : Dict[str, Any]
            Must contain 'preprocessed_data' and
            '<model_step_id>_model'.

        Returns
        -------
        Dict[str, Any]
            Context with predictions added under
            key '<step_id>_predictions'.

        Raises
        ------
        RuntimeError
            If required inputs missing from context.
        ValueError
            If model_step_id not specified.
        """
        self.validate_dependencies(context)

        if self.model_step_id is None:
            raise ValueError(
                f"Step '{self.step_id}' requires 'model_step_id' parameter"
            )

        if "preprocessed_data" not in context:
            raise RuntimeError(
                f"Step '{self.step_id}' requires 'preprocessed_data' " f"in context"
            )

        model_key = f"{self.model_step_id}_model"
        if model_key not in context:
            raise RuntimeError(
                f"Step '{self.step_id}' requires '{model_key}' in context"
            )

        df: pd.DataFrame = context["preprocessed_data"]
        model = context[model_key]

        # Extract features (exclude target and metadata columns)
        exclude_cols = ["target", "date", "dataset", "timestamp"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        if len(feature_cols) == 0:
            raise RuntimeError(
                f"Step '{self.step_id}': No feature columns found "
                f"after excluding {exclude_cols}"
            )

        X = df[feature_cols].values

        # Run inference
        predictions = model.predict(X)

        # Flatten if needed (some models return 2D arrays)
        if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
            if predictions.shape[1] == 1:
                predictions = predictions.flatten()

        # Store in context
        context[f"{self.step_id}_predictions"] = predictions

        # Optionally save to CSV
        if self.save_path:
            self._save_predictions(df, predictions, feature_cols)

        print(f"[{self.step_id}] Generated {len(predictions)} predictions")
        return context

    def _save_predictions(
        self, df: pd.DataFrame, predictions: np.ndarray, feature_cols: list
    ) -> None:
        """Save predictions to CSV file.

        Parameters
        ----------
        df : pd.DataFrame
            Original dataframe with input features.
        predictions : np.ndarray
            Model predictions.
        feature_cols : list
            List of feature column names.
        """
        output_data = {"prediction": predictions}

        # Optionally include input features
        if self.include_inputs:
            for col in feature_cols:
                output_data[col] = df[col].values

        pred_df = pd.DataFrame(output_data)

        # Create output directory if needed
        output_path = Path(self.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pred_df.to_csv(output_path, index=False)

        print(
            f"[{self.step_id}] Saved predictions to: {self.save_path} "
            f"({len(pred_df)} rows)"
        )
