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
    DEFAULT_OUTPUTS = {"features": "preprocessed_data", "targets": "target_data"}

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
        df = fea_df.copy()
        for col in target_cols:
            df[col] = df_raw[col]
        return df

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

    def execute_train(
        self,
        context: Dict[str, Any],
        df_full: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        TRAIN flow for offline preprocessor.

        - Fit the offline preprocessor on the training subset.
        - Transform the full training dataset.
        - Register the fitted preprocessor artifact if enabled.

        Returns
        -------
        pd.DataFrame
            DataFrame with preprocessed training data.
        """
        print(f"[{self.step_id}] Train flow - fitting offline preprocessor.")

        preprocessor = OfflinePreprocessor(is_train=True, cfg=self.cfg)

        if "dataset" in df_full.columns:
            fit_subset = train_df
        else:
            fit_subset = preprocessor.select_train_subset(df_full)

        preprocessor.fit_manager(fit_subset)
        df_transformed: pd.DataFrame = preprocessor.transform(df_full)

        context["preprocessor"] = preprocessor

        if self.log_artifact:
            self.register_for_discovery(context, preprocessor)

        return df_transformed

    def execute_eval(
        self,
        context: Dict[str, Any],
        df_full: pd.DataFrame,
        df_test: pd.DataFrame,
        is_split_input: bool,
    ) -> pd.DataFrame:
        """
        EVAL flow for offline preprocessor.

        - Restore a fitted transform manager from MLflow.
        - Wrap it into offline preprocessor without refitting.
        - Transform test data (upstream split or full input if not split).

        Returns
        -------
        pd.DataFrame
            DataFrame with transformed test data.
        """
        print(f"[{self.step_id}] Eval flow - restoring from MLflow.")

        transform_manager = self._load_preprocessor_from_mlflow()

        preprocessor = OfflinePreprocessor(is_train=False, cfg=self.cfg)
        preprocessor.transform_manager = transform_manager

        if not is_split_input:
            df_test = df_full.copy()

        df_transformed: pd.DataFrame = preprocessor.transform(df_test)

        if is_split_input:
            df_transformed["dataset"] = "test"
            print(f"[{self.step_id}] Upstream split used -> labeled as test.")
        else:
            print(f"[{self.step_id}] No split detected -> full input used as test.")

        context["preprocessor"] = preprocessor

        return df_transformed

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate TRAIN or EVAL flow and merge results into pipeline context.

        This method delegates execution to the corresponding flow-specific
        functions: execute_train() or execute_eval().

        Returns
        -------
        Dict[str, Any]
            Updated pipeline context with wired outputs.
        """
        self.validate_dependencies(context)

        df_full: pd.DataFrame = self.get_input(context, "df")

        train_df_result = self.get_input(context, "train_df", required=False)
        train_df: pd.DataFrame = (
            train_df_result if train_df_result is not None else pd.DataFrame()
        )

        test_df_result = self.get_input(context, "test_df", required=False)
        test_df: pd.DataFrame = (
            test_df_result if test_df_result is not None else pd.DataFrame()
        )

        is_split_input: bool = bool(context.get("is_splited_input", False))

        data_cfg: Dict[str, Any] = self.cfg.get("data", {})
        target_columns: List[str] = data_cfg.get("target_columns", [])
        feature_columns: List[str] = data_cfg.get("features", [])

        if self.is_train:
            df_transformed = self.execute_train(context, df_full, train_df)
        else:
            df_transformed = self.execute_eval(
                context, df_full, test_df, is_split_input
            )

        self.set_output(
            context,
            "features",
            df_transformed[feature_columns],
            "preprocessed_data",
        )

        self.set_output(
            context,
            "targets",
            df_transformed[target_columns],
            "target_data",
        )

        return context


# Register step type
StepFactory.register("preprocessor", PreprocessorStep)
