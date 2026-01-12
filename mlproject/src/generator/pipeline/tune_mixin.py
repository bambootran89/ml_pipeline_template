from __future__ import annotations

import copy
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from .base_transform_mixin import BaseTransformMixin


class TuneMixin(BaseTransformMixin):
    """Mixin providing pipeline transformation logic for hyperparameter tuning.

    This mixin is responsible for converting a training pipeline into a
    tuning-aware pipeline by inserting tuner steps and rewiring model
    producers to consume tuned hyperparameters.

    Core responsibilities:
    - Insert a dedicated tuner step immediately before each model producer.
    - Modify model producer steps so they explicitly depend on the
      corresponding tuner and consume tuned parameters
      (`use_tuned_params=True`, `tune_step_id` set).
    - Preserve original execution order and dependencies as much as possible,
      only extending them where tuning requires it.
    - Recursively handle complex pipeline structures, including:
      * Sub-pipelines (nested pipelines).
      * Parallel branches.
      * Conditional branches (`if_true` / `if_false`).
    - Maintain a mapping between model producer IDs and their associated
      tuner step IDs to ensure consistent rewiring.

    Design constraints:
    - No training or tuning logic is changed; this mixin only restructures
      the pipeline configuration.
    - Original step semantics, ordering, and non-model steps remain intact.
    - The mixin assumes the host class provides `_is_model_producer`.

    The output of this mixin is a valid pipeline definition that can be
    executed by a tuning engine (e.g. Optuna-based) while remaining fully
    compatible with the original training pipeline structure.
    """

    def _create_tuner_step(self, model_producer: Any) -> DictConfig:
        """Create a tuner step for a model producer.

        Args:
            model_producer: The model producer step to create tuner for.

        Returns:
            DictConfig for the tuner step.
        """
        tuner_id = f"tune_{model_producer.id}"
        depends_on = (
            model_producer.depends_on if hasattr(model_producer, "depends_on") else []
        )
        tuner_config = {
            "id": tuner_id,
            "type": "tuner",
            "enabled": True,
            "depends_on": depends_on,
            "target_model_id": model_producer.id,  # Specify which model to tune
        }
        return OmegaConf.create(tuner_config)

    def _modify_model_producer_for_tuning(self, step: Any, tuner_id: str) -> Any:
        """Modify model producer step to use tuned hyperparameters.

        Args:
            step: The model producer step to modify.
            tuner_id: ID of the corresponding tuner step.

        Returns:
            Modified step with tuning configuration.
        """
        modified = copy.deepcopy(step)
        modified.use_tuned_params = True
        modified.tune_step_id = tuner_id

        # Update depends_on to include tuner
        if not hasattr(modified, "depends_on"):
            modified.depends_on = []
        if tuner_id not in modified.depends_on:
            modified.depends_on.insert(0, tuner_id)

        return modified

    def _process_step_for_tuning(
        self,
        step: Any,
        new_steps: List[Any],
        tuner_map: Dict[str, str],
    ) -> None:
        """Process a single step for tune pipeline generation.

        Args:
            step: Step to process.
            new_steps: List to append processed steps to.
            tuner_map: Mapping of model_producer_id -> tuner_id.
        """
        if self._is_model_producer(step):
            self._process_model_producer_step(step, new_steps, tuner_map)
            return

        step_type = step.type

        if step_type == "sub_pipeline":
            self._process_sub_pipeline_step(step, new_steps, tuner_map)
            return

        if step_type == "parallel":
            self._process_parallel_step(step, new_steps, tuner_map)
            return

        if step_type == "branch":
            self._process_branch_step(step, new_steps, tuner_map)
            return

        new_steps.append(copy.deepcopy(step))

    def _process_model_producer_step(
        self,
        step: Any,
        new_steps: List[Any],
        tuner_map: Dict[str, str],
    ) -> None:
        """Add tuner before model producer and modify producer."""
        tuner_step = self._create_tuner_step(step)
        new_steps.append(tuner_step)

        tuner_map[step.id] = tuner_step.id

        modified_producer = self._modify_model_producer_for_tuning(
            step,
            tuner_step.id,
        )
        new_steps.append(modified_producer)

    def _process_sub_pipeline_step(
        self,
        step: Any,
        new_steps: List[Any],
        tuner_map: Dict[str, str],
    ) -> None:
        """Recursively process sub-pipeline steps."""
        modified_sub = copy.deepcopy(step)

        if hasattr(modified_sub, "pipeline") and hasattr(
            modified_sub.pipeline,
            "steps",
        ):
            nested_steps: List[Any] = []
            for nested_step in modified_sub.pipeline.steps:
                self._process_step_for_tuning(
                    nested_step,
                    nested_steps,
                    tuner_map,
                )
            modified_sub.pipeline.steps = nested_steps

        new_steps.append(modified_sub)

    def _process_parallel_step(
        self,
        step: Any,
        new_steps: List[Any],
        tuner_map: Dict[str, str],
    ) -> None:
        """Process parallel branches for tuning."""
        modified_parallel = copy.deepcopy(step)

        if hasattr(modified_parallel, "branches"):
            new_branches: List[Any] = []
            for branch in modified_parallel.branches:
                if self._is_model_producer(branch):
                    tuner_step = self._create_tuner_step(branch)
                    new_steps.append(tuner_step)

                    tuner_map[branch.id] = tuner_step.id

                    modified_branch = self._modify_model_producer_for_tuning(
                        branch,
                        tuner_step.id,
                    )
                    new_branches.append(modified_branch)
                else:
                    new_branches.append(branch)

            modified_parallel.branches = new_branches

        new_steps.append(modified_parallel)

    def _process_branch_step(
        self,
        step: Any,
        new_steps: List[Any],
        tuner_map: Dict[str, str],
    ) -> None:
        """Process conditional branch steps for tuning."""
        modified_branch_step = copy.deepcopy(step)

        for branch_name in ("if_true", "if_false"):
            if not hasattr(modified_branch_step, branch_name):
                continue

            branch = getattr(modified_branch_step, branch_name)
            if not self._is_model_producer(branch):
                continue

            tuner_step = self._create_tuner_step(branch)
            new_steps.append(tuner_step)

            tuner_map[branch.id] = tuner_step.id

            modified_branch = self._modify_model_producer_for_tuning(
                branch,
                tuner_step.id,
            )
            setattr(modified_branch_step, branch_name, modified_branch)

        new_steps.append(modified_branch_step)

    def _build_tune_steps(self, train_steps: List[Any]) -> List[Any]:
        """Build steps for tune pipeline.

        Args:
            train_steps: Original training pipeline steps.

        Returns:
            List of steps for tune pipeline with tuners added.
        """
        new_steps: List[Any] = []
        tuner_map: Dict[str, str] = {}  # model_id -> tuner_id mapping

        for step in train_steps:
            self._process_step_for_tuning(step, new_steps, tuner_map)

        return new_steps
