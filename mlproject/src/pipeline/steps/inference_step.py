"""Inference step for test/prediction pipelines.

This module provides a step that generates predictions without
evaluation metrics, useful for production inference.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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

        # --- Handle timeseries vs tabular separately ---
        data_type: str = self.cfg.data.get("type", "timeseries")
        if data_type == "timeseries":
            entity_key: str = self.cfg.data.get("entity_key", "location_id")
            win: int = int(
                self.cfg.experiment.get("hyperparams", {}).get("input_chunk_length", 24)
            )
            if "entity_key" in df.columns:
                arr_list: List[np.ndarray] = [
                    self._prepare_input_window(
                        g.drop(columns=[entity_key], errors="ignore"),
                        win,
                    )
                    for _, g in df.groupby(entity_key)
                ]
                print(f"[INFERENCE] Building input window of length {win}")

                x = np.vstack(arr_list).astype(np.float32)

                print(f"[INFERENCE] Input window shape: {x.shape}")
            else:
                x = self._prepare_input_window(
                    df,
                    win,
                ).astype(np.float32)

        else:
            # For tabular, no sequence window; use full preprocessed DataFrame
            print("[INFERENCE] Tabular input, using full DataFrame")
            x = df.values.astype(np.float32)
            print(f"[INFERENCE] Input array shape: {x.shape}")

        print("[INFERENCE] Running model.predict()")
        y: np.ndarray = model.predict(x)
        print(f"[INFERENCE] Output shape: {y.shape}")

        if hasattr(y, "flatten"):
            print(f"[INFERENCE] First 10 output values: {y[:10]}")

        print("[INFERENCE] Inference completed")
        # Store in context
        context[f"{self.step_id}_predictions"] = y

        return context

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """
        Build model input window for prediction.

        Args:
            df: Preprocessed DataFrame.
            input_chunk_length: Sequence length required by model.

        Returns:
            Model input array with shape [1, seq_len, n_features].

        Raises:
            ValueError: If input has fewer rows than input_chunk_length.
        """
        seq_len: int = input_chunk_length
        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")
        window: np.ndarray = df.iloc[-seq_len:].values
        return window[np.newaxis, :].astype(np.float32)
