"""Enhanced inference step with multi-source feature support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.pipeline.steps.base import PipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class InferenceStep(PipelineStep):
    """Generate predictions with multi-source feature composition.

    This step composes features from multiple sources before inference,
    enabling models trained on engineered features to make predictions.

    Context Inputs (configurable via wiring)
    -----------------------------------------
    features : pd.DataFrame or array-like
        Primary feature source.
    additional_feature_keys : List[str], optional
        Additional feature source keys to compose.
    model : Any
        Trained model for inference.

    Context Outputs
    ---------------
    predictions : np.ndarray
        Model predictions.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        include_inputs: bool = False,
        additional_feature_keys: Optional[List[str]] = None,
        feature_align_method: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize enhanced inference step.

        Parameters
        ----------
        step_id : str
            Unique step identifier.
        cfg : Any
            Configuration object.
        enabled : bool
            Whether step should execute.
        depends_on : Optional[List[str]]
            Prerequisite steps.
        save_path : Optional[str]
            Path to save predictions.
        include_inputs : bool
            Whether to save input features with predictions.
        additional_feature_keys : Optional[List[str]]
            Keys for additional feature sources.
        feature_align_method : str
            Method for aligning feature shapes.
        **kwargs
            Additional parameters.
        """
        super().__init__(
            step_id,
            cfg,
            enabled,
            depends_on,
            additional_feature_keys=additional_feature_keys,
            feature_align_method=feature_align_method,
            **kwargs,
        )
        self.save_path = save_path
        self.include_inputs = include_inputs

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with predictions added.
        """
        self.validate_dependencies(context)

        # Compose features from multiple sources
        features_df, _ = self.get_composed_features(context, "features", required=True)

        # Get model
        model = self.get_input(context, "model")

        # Prepare input
        x = self._prepare_model_input(features_df)

        # Run inference
        print(f"[{self.step_id}] Running inference on shape {x.shape}")
        predictions = model.predict(x)

        # Store output - always store to predictions
        self.set_output(context, "predictions", predictions)

        # Also store to features if configured (for clustering/feature-generating steps)
        # This ensures clustering models that output_as_feature work correctly in
        # serve mode
        if "features" in self.output_keys:
            self.set_output(context, "features", predictions)
            print(
                f"[{self.step_id}] Also stored to features key: "
                f"{self.output_keys['features']}"
            )

        # Optional: Save predictions
        if self.save_path:
            self._save_predictions(
                predictions, features_df if self.include_inputs else None
            )

        print(
            f"[{self.step_id}] Inference complete. "
            f"Output shape: {predictions.shape}"
        )

        return context

    def _prepare_model_input(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare input array for model.

        Parameters
        ----------
        features_df : pd.DataFrame
            Composed features.

        Returns
        -------
        np.ndarray
            Model input array.
        """
        data_type = self.cfg.data.get("type", "timeseries")

        if data_type == "timeseries":
            return self._prepare_timeseries_input(features_df)

        # Tabular data: use as-is
        return features_df.values.astype(np.float32)

    def _prepare_timeseries_input(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Prepare timeseries windowed input.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        np.ndarray
            Windowed input array.
        """
        entity_key = self.cfg.data.get("entity_key", "location_id")
        hyperparams = self.cfg.experiment.get("hyperparams", {})
        win_in = int(hyperparams.get("input_chunk_length", 24))
        win_out = int(hyperparams.get("out_chunk_length", 6))

        if entity_key in df.columns:
            windows = []
            for _, group in df.groupby(entity_key):
                group_clean = group.drop(columns=[entity_key], errors="ignore")
                windows.append(self._build_windows(group_clean, win_in, win_out))
            return np.vstack(windows).astype(np.float32)

        return self._build_windows(df, win_in, win_out).astype(np.float32)

    def _build_windows(
        self,
        df: pd.DataFrame,
        input_length: int,
        output_length: int,
    ) -> np.ndarray:
        """Build sliding windows from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        input_length : int
            Window input length.
        output_length : int
            Window output length (stride).

        Returns
        -------
        np.ndarray
            Windowed array [n_windows, input_length, n_features].
        """
        n_rows = len(df)
        if n_rows < input_length:
            raise ValueError(f"Need at least {input_length} rows, got {n_rows}")

        windows = []
        for start in range(0, n_rows - input_length + 1, output_length):
            window = df.iloc[start : start + input_length].values
            windows.append(window)

        return np.array(windows, dtype=np.float32)

    def _save_predictions(
        self,
        predictions: np.ndarray,
        inputs: Optional[pd.DataFrame],
    ) -> None:
        """Save predictions to file.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions.
        inputs : Optional[pd.DataFrame]
            Input features if include_inputs=True.
        """
        df = pd.DataFrame(predictions, columns=["prediction"])

        if inputs is not None:
            df = pd.concat([inputs.reset_index(drop=True), df], axis=1)

        df.to_csv(self.save_path, index=False)
        print(f"[{self.step_id}] Saved predictions to {self.save_path}")


StepFactory.register("inference", InferenceStep)
