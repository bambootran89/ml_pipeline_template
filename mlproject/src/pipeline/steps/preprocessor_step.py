"""Preprocessing step with data wiring support.

Supports loading preprocessor from MLflow in eval/serve mode.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.preprocess.offline import OfflinePreprocessor
from mlproject.src.tracking.mlflow_manager import MLflowManager


class PreprocessorStep(BasePipelineStep):
    """Fit and apply preprocessing transformations.

    This step fits preprocessing on training data and transforms
    the full dataset. Supports data wiring for flexible I/O.

    In TRAIN mode (is_train=True):
    - Fits preprocessor on training subset
    - Transforms full dataset
    - Stores preprocessor in context for logging

    In EVAL/SERVE mode (is_train=False):
    - Loads preprocessor from MLflow Registry
    - Transforms test/input data

    Context Inputs (configurable via wiring)
    -----------------------------------------
    df : pd.DataFrame
        Full input data (required).
    train_df : pd.DataFrame
        Training subset (optional).
    test_df : pd.DataFrame
        Test subset (optional).

    Context Outputs (configurable via wiring)
    ------------------------------------------
    preprocessed_data : pd.DataFrame
        Transformed feature data (default key).
    preprocessor : OfflinePreprocessor
        Fitted/loaded preprocessor instance.

    Configuration
    -------------
    is_train : bool, default=True
        If True, fit preprocessor on training data.
        If False, load from MLflow Registry.
    alias : str, default="latest"
        MLflow model alias for loading (only used when is_train=False).
    """

    DEFAULT_INPUTS = {"df": "df", "train_df": "train_df", "test_df": "test_df"}
    DEFAULT_OUTPUTS = {"data": "preprocessed_data"}

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        is_train: bool = True,
        alias: str = "latest",
        **kwargs: Any,
    ) -> None:
        """Initialize preprocessing step.

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
        is_train : bool, default=True
            Whether to fit (train mode) or load from MLflow (eval mode).
        alias : str, default="latest"
            MLflow alias for loading preprocessor (latest/production/staging).
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.is_train = is_train
        self.alias = alias

    def _attach_targets_if_needed(
        self, df_raw: pd.DataFrame, fea_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach target columns back for tabular datasets.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw input data.
        fea_df : pd.DataFrame
            Transformed features.

        Returns
        -------
        pd.DataFrame
            Final dataset for evaluation.
        """
        data_cfg = self.cfg.get("data", {})
        data_type = str(data_cfg.get("type", "timeseries")).lower()

        if data_type == "timeseries":
            return fea_df

        target_cols = data_cfg.get("target_columns", [])
        if not target_cols:
            return fea_df

        tar_df = df_raw[target_cols]
        return pd.concat([fea_df, tar_df], axis=1)

    def _load_preprocessor_from_mlflow(self) -> Any:
        """Load preprocessor (transform_manager) from MLflow Registry.

        Returns
        -------
        Any
            Loaded transform_manager object.

        Raises
        ------
        RuntimeError
            If MLflow disabled or preprocessor not found.
        """
        mlflow_manager = MLflowManager(self.cfg)

        if not mlflow_manager.enabled:
            raise RuntimeError(
                f"Step '{self.step_id}': MLflow must be enabled to load "
                f"preprocessor in eval mode. Set mlflow.enabled=true or "
                f"use is_train=true to fit locally."
            )

        experiment_name = self.cfg.experiment.get("name", "")
        if not experiment_name:
            raise ValueError(
                f"Step '{self.step_id}': experiment.name must be specified "
                f"in config to load preprocessor from MLflow."
            )

        registry_name = f"{experiment_name}_preprocessor"

        print(
            f"[{self.step_id}] Loading preprocessor from MLflow: "
            f"name='{registry_name}', alias='{self.alias}'"
        )

        transform_manager = mlflow_manager.load_component(
            name=registry_name,
            alias=self.alias,
        )

        if transform_manager is None:
            raise RuntimeError(
                f"Failed to load preprocessor '{registry_name}' with alias "
                f"'{self.alias}'. Make sure it was logged during training. "
                f"Run training pipeline first to create preprocessor artifacts."
            )

        print(f"[{self.step_id}] Successfully loaded preprocessor from MLflow")
        return transform_manager

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fit preprocessor and transform data.

        In TRAIN mode: Fit on training data, transform full dataset.
        In EVAL mode: Load from MLflow, transform test data.

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Context with preprocessed data and preprocessor added.
        """
        self.validate_dependencies(context)

        # Get inputs using wiring
        df: pd.DataFrame = self.get_input(context, "df")

        # Handle optional DataFrames - can't use `or` with DataFrame
        train_df_result = self.get_input(context, "train_df", required=False)
        train_df: pd.DataFrame = (
            train_df_result if train_df_result is not None else pd.DataFrame()
        )

        test_df_result = self.get_input(context, "test_df", required=False)
        test_df: pd.DataFrame = (
            test_df_result if test_df_result is not None else pd.DataFrame()
        )

        is_splited = context.get("is_splited_input", False)

        if self.is_train:
            # ========================================
            # TRAINING MODE: Fit preprocessor locally
            # ========================================
            print(f"[{self.step_id}] Training mode - fitting preprocessor")

            preprocessor = OfflinePreprocessor(is_train=True, cfg=self.cfg)

            # Determine training subset
            if "dataset" in df.columns:
                train_subset = train_df
            else:
                train_subset = preprocessor.select_train_subset(df)

            # Fit and transform
            preprocessor.fit_manager(train_subset)
            df_transformed = preprocessor.transform(df)

            # Store preprocessor for later logging
            context["preprocessor"] = preprocessor

        else:
            # ========================================
            # EVAL/SERVE MODE: Load from MLflow
            # ========================================
            print(f"[{self.step_id}] Eval mode - loading preprocessor from MLflow")

            # Load transform_manager from MLflow
            transform_manager = self._load_preprocessor_from_mlflow()

            # Create preprocessor wrapper with loaded transform_manager
            preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
            preprocessor.transform_manager = transform_manager

            # Determine data to transform
            if not is_splited:
                test_df = df.copy()

            # Transform
            df_transformed = preprocessor.transform(test_df)
            df_transformed = self._attach_targets_if_needed(test_df, df_transformed)

            # Log split status
            if is_splited:
                df_transformed["dataset"] = "test"
                print(
                    f"[{self.step_id}] Test split already performed upstream "
                    "-> assigning dataset='test' label."
                )
            else:
                print(
                    f"[{self.step_id}] Using full input as test data "
                    "(no pre-split detected)."
                )

            # Store for reference (already fitted, won't be logged again)
            context["preprocessor"] = preprocessor

        # Store outputs using wiring
        self.set_output(context, "data", df_transformed, "preprocessed_data")

        print(f"[{self.step_id}] Preprocessed data: {df_transformed.shape}")
        return context


# Register step type
StepFactory.register("preprocessor", PreprocessorStep)
