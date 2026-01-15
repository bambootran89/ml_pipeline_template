"""Enhanced DataModule step with multi-source feature support."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory


class DataModuleStep(BasePipelineStep):
    """Build DataModule with optional multi-source feature composition.

    Context Inputs
    --------------
    features : pd.DataFrame
        Base features from preprocessing.
    targets : pd.DataFrame
        Target labels.
    additional_feature_keys : List[str], optional
        Additional feature sources to compose.

    Context Outputs
    ---------------
    datamodule : Any
        Constructed DataModule instance.
    """

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        output_as_feature: bool = False,
        additional_feature_keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DataModule step.

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
        output_as_feature : bool
            Whether to generate features from model predictions.
        additional_feature_keys : Optional[List[str]]
            Keys for additional feature sources to compose.
        **kwargs
            Additional parameters.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.output_as_feature = output_as_feature
        self.additional_feature_keys = additional_feature_keys or []

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DataModule construction with feature composition.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Updated context with DataModule.
        """
        self.validate_dependencies(context)

        # Get base features and targets
        df_features = self.get_input(context, "features")
        df_targets = self.get_input(context, "targets", required=False)

        # Compose with additional features if specified
        if self.additional_feature_keys:
            df_features = self._compose_features(context, df_features)

        # Build input DataFrame
        data_cfg: Dict[str, Any] = self.cfg.get("data", {})
        data_type: str = str(data_cfg.get("type", "tabular")).lower()

        if data_type == "timeseries":
            input_df = df_features.copy()
        elif df_targets is not None:
            input_df = pd.concat([df_features, df_targets], axis=1)
        else:
            input_df = df_features.copy()

        # Build DataModule
        print(f"[{self.step_id}] Building DataModule with shape {input_df.shape}")
        dm = DataModuleFactory.build(self.cfg, input_df)
        dm.setup()

        self.set_output(context, "datamodule", dm)

        # Optional: Generate features from model
        if self.output_as_feature:
            self._generate_model_features(context, dm)

        return context

    def _compose_features(
        self,
        context: Dict[str, Any],
        base_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compose base features with additional sources.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        base_features : pd.DataFrame
            Base feature DataFrame.

        Returns
        -------
        pd.DataFrame
            Composed features.
        """
        if not isinstance(base_features, pd.DataFrame):
            base_features = pd.DataFrame(base_features)

        composed = base_features.copy()
        n_samples = len(composed)

        print(f"[{self.step_id}] Composing features:")
        print(f"  Base: {composed.shape}")

        for key in self.additional_feature_keys:
            additional = context.get(key)
            if additional is None:
                print(f"  Warning: Feature key '{key}' not found, skipping")
                continue

            # Convert to DataFrame
            if isinstance(additional, np.ndarray):
                additional = self._ndarray_to_df(additional, key)
            elif not isinstance(additional, pd.DataFrame):
                additional = pd.DataFrame(additional)

            # Align samples
            additional = self._align_samples(additional, n_samples, key)

            # Force index alignment if lengths match
            # This handles cases where additional features lost their index (e.g. from numpy)
            if len(additional) == len(composed):
                additional.index = composed.index

            # Prefix columns to avoid collisions
            if isinstance(additional, pd.DataFrame):
                additional.columns = [f"{key}_{c}" for c in additional.columns]

            # Concatenate
            composed = pd.concat([composed, additional], axis=1)
            print(f"  + {key}: {additional.shape} -> Total: {composed.shape}")

        return composed

    def _ndarray_to_df(self, arr: np.ndarray, prefix: str) -> pd.DataFrame:
        """Convert numpy array to DataFrame.

        Parameters
        ----------
        arr : np.ndarray
            Input array.
        prefix : str
            Column name prefix.

        Returns
        -------
        pd.DataFrame
            Converted DataFrame.
        """
        if arr.ndim == 1:
            return pd.DataFrame({f"{prefix}_0": arr})

        if arr.ndim == 2:
            cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols)

        raise ValueError(
            f"Cannot convert {arr.ndim}D array to DataFrame. " "Supported: 1D, 2D"
        )

    def _align_samples(
        self,
        df: pd.DataFrame,
        target_samples: int,
        key: str,
    ) -> pd.DataFrame:
        """Align DataFrame to target number of samples.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target_samples : int
            Target number of samples.
        key : str
            Feature key (for error messages).

        Returns
        -------
        pd.DataFrame
            Aligned DataFrame.

        Raises
        ------
        ValueError
            If shapes cannot be aligned.
        """
        current = len(df)

        if current == target_samples:
            return df

        # Broadcast: single sample to multiple
        if current == 1:
            return pd.concat([df] * target_samples, ignore_index=True)

        # Repeat: tile to match
        if current < target_samples and target_samples % current == 0:
            n_repeats = target_samples // current
            return pd.concat([df] * n_repeats, ignore_index=True)

        # Truncate: cut to size
        if current > target_samples:
            return df.iloc[:target_samples]

        raise ValueError(
            f"Cannot align feature '{key}' with {current} samples "
            f"to {target_samples} samples. Shapes incompatible."
        )

    def _generate_model_features(
        self,
        context: Dict[str, Any],
        datamodule: Any,
    ) -> None:
        """Generate features from model predictions.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.
        datamodule : Any
            DataModule instance.
        """
        model = context.get("model")
        if model is None:
            print(
                f"[{self.step_id}] Warning: output_as_feature=True "
                "but no model found"
            )
            return

        # Get data for prediction
        if hasattr(datamodule, "get_test_windows"):
            x_data, _ = datamodule.get_test_windows()
        else:
            x_data = datamodule.get_data()[-2]

        # Generate predictions
        preds = model.predict(x_data)
        features = np.asarray(preds, dtype=float)

        self.set_output(context, "features", features)
        print(f"[{self.step_id}] Generated features from model: " f"{features.shape}")


StepFactory.register("datamodule", DataModuleStep)
