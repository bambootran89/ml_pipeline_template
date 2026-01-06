"""Model training step with data wiring support.

Enhanced to support using best params from TuningStep.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import OmegaConf

from mlproject.src.datamodule.factory import DataModuleFactory
from mlproject.src.models.model_factory import ModelFactory
from mlproject.src.pipeline.steps.base import BasePipelineStep
from mlproject.src.pipeline.steps.factory_step import StepFactory
from mlproject.src.trainer.factory import TrainerFactory


class TrainerStep(BasePipelineStep):
    """Train a model on preprocessed data.

    This step supports:
    - Data wiring for flexible input/output keys
    - Multiple model types (ML/DL)
    - Using tuned hyperparameters from TuningStep
    - Saving trained artifacts

    Context Inputs (configurable via wiring)
    -----------------------------------------
    data : pd.DataFrame
        Transformed features (default: preprocessed_data).
    <dep_id>_features : np.ndarray, optional
        Injected features from dependency steps.
    <tune_step_id>_best_params : Dict, optional
        Best hyperparameters from TuningStep.

    Context Outputs (configurable via wiring)
    ------------------------------------------
    model : Any
        Trained model wrapper.
    datamodule : DataModule
        Built datamodule.
    features : np.ndarray, optional
        Model predictions if output_as_feature=True.

    Wiring Example
    --------------
    ::

        - id: "train_xgb"
          type: "trainer"
          depends_on: ["preprocess", "kmeans"]
          wiring:
            inputs:
              data: "custom_features"
            outputs:
              model: "xgb_model"
              features: "xgb_predictions"
          output_as_feature: true

    Configuration
    -------------
    output_as_feature : bool, default=False
        If True, store predictions for downstream injection.
    use_tuned_params : bool, default=False
        If True, use best_params from upstream TuningStep.
    tune_step_id : str, optional
        ID of TuningStep (required if use_tuned_params=True).
    """

    DEFAULT_INPUTS = {"data": "preprocessed_data"}

    def __init__(
        self,
        step_id: str,
        cfg: Any,
        enabled: bool = True,
        depends_on: Optional[List[str]] = None,
        use_tuned_params: bool = False,
        tune_step_id: Optional[str] = None,
        output_as_feature: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize model training step.

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
        use_tuned_params : bool, default=False
            Whether to use best params from TuningStep.
        tune_step_id : str, optional
            ID of TuningStep to read best_params from.
        output_as_feature : bool, default=False
            If True, store predictions for downstream steps.
        **kwargs
            Additional parameters including wiring config.
        """
        super().__init__(step_id, cfg, enabled, depends_on, **kwargs)
        self.output_as_feature = output_as_feature
        self.use_tuned_params = use_tuned_params
        self.tune_step_id = tune_step_id

    def _get_hyperparams(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get hyperparameters (from config or tuning).

        Parameters
        ----------
        context : Dict[str, Any]
            Pipeline context.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters to use for training.
        """
        if self.use_tuned_params:
            if self.tune_step_id is None:
                raise ValueError(
                    f"Step '{self.step_id}': use_tuned_params=True requires "
                    f"tune_step_id parameter"
                )

            best_params_key = f"{self.tune_step_id}_best_params"
            if best_params_key not in context:
                raise ValueError(
                    f"Step '{self.step_id}': Expected '{best_params_key}' "
                    f"in context. Make sure TuningStep runs before this step."
                )

            best_params = context[best_params_key]

            print(f"\n[{self.step_id}] Using tuned hyperparameters:")
            for param, value in best_params.items():
                print(f"  - {param}: {value}")

            cfg_copy = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            if "args" not in cfg_copy.experiment.hyperparams:
                cfg_copy.experiment.hyperparams.args = {}

            for param, value in best_params.items():
                cfg_copy.experiment.hyperparams.args[param] = value

            return dict(cfg_copy.experiment.hyperparams)
        else:
            return dict(self.cfg.experiment.hyperparams)

    def _prepare_data(
        self,
        context: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Prepare the dataset for training based on experiment configuration.

        - In timeseries mode, use features only.
        - Otherwise, concatenate features and targets.

        Returns
        -------
        pd.DataFrame
            DataFrame ready for DataModule setup.
        """
        df_features: pd.DataFrame = self.get_input(context, "features")
        df_targets: pd.DataFrame = self.get_input(context, "targets")

        data_cfg: Dict[str, Any] = self.cfg.get("data", {})
        data_type: str = str(data_cfg.get("type", "tabular")).lower()

        if data_type == "timeseries":
            return df_features.copy()

        return pd.concat([df_features, df_targets], axis=1)

    def _train_model(
        self,
        context: Dict[str, Any],
        train_df: pd.DataFrame,
    ) -> tuple[Any, Any]:
        """
        Create pipeline components and train the model.

        This function builds the model wrapper, datamodule, and trainer,
        then executes the training process.

        Returns
        -------
        tuple[Any, Any]
            Trained model wrapper and initialized datamodule.
        """
        model_name: str = self.cfg.experiment.model.lower()
        model_type: str = self.cfg.experiment.model_type.lower()
        model_class = ModelFactory.create(model_name, self.cfg)
        data_module = DataModuleFactory.build(self.cfg, train_df)
        data_module.setup()

        trainer = TrainerFactory.create(
            model_type=model_type,
            model_name=model_name,
            wrapper=model_class,
            save_dir=self.cfg.training.artifacts_dir,
        )

        print(f"[{self.step_id}] Starting model training.")
        trained_wrapper = trainer.train(data_module, self._get_hyperparams(context))

        return trained_wrapper, data_module

    def execute(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Coordinate model training and register outputs into pipeline context.

        This method delegates:
        - Data preparation to _prepare_data().
        - Model training to _train_model().

        Returns
        -------
        Dict[str, Any]
            Updated pipeline context containing trained model and wired outputs.
        """
        self.validate_dependencies(context)

        train_df = self._prepare_data(context)
        trained_wrapper, data_module = self._train_model(context, train_df)

        self.set_output(context, "model", trained_wrapper)
        self.set_output(context, "datamodule", data_module)

        if self.log_artifact:
            self.register_for_discovery(context, trained_wrapper)

        if self.output_as_feature:
            x_train = (
                data_module.get_data()[0]
                if hasattr(data_module, "get_data")
                else data_module.get_test_windows()[0]
            )
            preds = trained_wrapper.predict(x_train)
            self.set_output(context, "features", preds)
            print(f"[{self.step_id}] Generated downstream features: {preds.shape}")

        print(f"[{self.step_id}] Model trained and registered successfully.")
        print(context)
        return context


# Register step type
StepFactory.register("trainer", TrainerStep)
