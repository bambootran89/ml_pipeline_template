"""Inference step with data wiring support.

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
    saves predictions to disk. Supports data wiring.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    data : pd.DataFrame
        Preprocessed features (default: preprocessed_data).
    model : ModelWrapper
        Loaded model instance.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    predictions : np.ndarray
        Raw model predictions.

    Wiring Example
    --------------
    ::

        - id: "inference"
          type: "inference"
          depends_on: ["load_model", "preprocess"]
          model_step_id: "load_model"
          wiring:
            inputs:
              data: "custom_features"
              model: "production_model"
            outputs:
              predictions: "final_predictions"
    """

    DEFAULT_INPUTS = {"data": "preprocessed_data"}

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        model_step_id: Optional[str] = None,
        save_path: Optional[str] = None,
        include_inputs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize inference step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : DictConfig
            Configuration object.
        enabled : bool, default=True
            Whether step is active.
        depends_on : Optional[List[str]], default=None
            Prerequisite steps.
        model_step_id : str, optional
            ID of step that loaded the model.
        save_path : str, optional
            Path to save predictions CSV.
        include_inputs : bool, default=False
            Whether to save input features with predictions.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.model_step_id = model_step_id
        self.save_path = save_path
        self.include_inputs = include_inputs

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference and optionally save predictions.

        Uses wiring configuration for input/output key mapping.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with predictions added.

        Raises
        ------
        ValueError
            If model_step_id not specified and model not in wiring.
        """
        self.validate_dependencies(context)

        # Get data using wiring
        df: pd.DataFrame = self.get_input(context, "data")

        # Get model using wiring or default pattern
        if self.model_step_id:
            model_key = f"{self.model_step_id}_model"
        else:
            model_key = None

        model = self.get_input(context, "model", default_key=model_key)

        # Handle timeseries vs tabular
        data_type: str = self.cfg.data.get("type", "timeseries")

        if data_type == "timeseries":
            entity_key: str = self.cfg.data.get("entity_key", "location_id")
            win: int = int(
                self.cfg.experiment.get("hyperparams", {}).get("input_chunk_length", 24)
            )

            if entity_key in df.columns:
                arr_list: List[np.ndarray] = [
                    self._prepare_input_window(
                        g.drop(columns=[entity_key], errors="ignore"), win
                    )
                    for _, g in df.groupby(entity_key)
                ]
                print(f"[{self.step_id}] Building input window of length {win}")
                x = np.vstack(arr_list).astype(np.float32)
                print(f"[{self.step_id}] Input window shape: {x.shape}")
            else:
                x = self._prepare_input_window(df, win).astype(np.float32)
        else:
            print(f"[{self.step_id}] Tabular input, using full DataFrame")
            x = df.values.astype(np.float32)
            print(f"[{self.step_id}] Input array shape: {x.shape}")

        print(f"[{self.step_id}] Running model.predict()")
        y: np.ndarray = model.predict(x)
        print(f"[{self.step_id}] Output shape: {y.shape}")

        if hasattr(y, "flatten") and len(y) > 0:
            print(f"[{self.step_id}] First 10 values: {y[:10]}")

        # Store output using wiring
        self.set_output(context, "predictions", y)

        print(f"[{self.step_id}] Inference completed")
        return context

    def _prepare_input_window(
        self, df: pd.DataFrame, input_chunk_length: int
    ) -> np.ndarray:
        """Build model input window for prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame.
        input_chunk_length : int
            Sequence length required by model.

        Returns
        -------
        np.ndarray
            Model input array with shape [1, seq_len, n_features].

        Raises
        ------
        ValueError
            If input has fewer rows than input_chunk_length.
        """
        seq_len: int = input_chunk_length
        if len(df) < seq_len:
            raise ValueError(f"Input data has {len(df)} rows, need at least {seq_len}")
        window: np.ndarray = df.iloc[-seq_len:].values
        return window[np.newaxis, :].astype(np.float32)
