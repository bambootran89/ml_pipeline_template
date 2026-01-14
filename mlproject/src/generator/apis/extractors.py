from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig


class ApiGeneratorExtractorsMixin:
    """Extraction helpers for API generator using registry patterns."""

    # Cấu hình các patterns cho model để tránh hardcode trong logic
    MODEL_PATTERNS: Dict[str, List[str]] = {
        "ml": [
            "xgboost",
            "xgb",
            "catboost",
            "kmeans",
            "kmean",
            "lightgbm",
            "lgbm",
            "randomforest",
            "rf",
        ],
        "dl": ["tft", "nlinear", "transformer", "lstm", "gru", "rnn"],
    }

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract inference steps from pipeline using a dispatcher map.

        Args:
            steps: List of pipeline steps.

        Returns:
            List of extracted inference information.
        """
        inference_steps: List[Dict[str, Any]] = []

        # Dispatcher map để thay thế if-elif
        handlers: Dict[str, Callable[[Any], List[Dict[str, Any]]]] = {
            "inference": lambda s: [self._extract_inference_info(s)],
            "branch": self._extract_branch_inferences,
            "sub_pipeline": self._extract_sub_pipeline_inferences,
        }

        for step in steps:
            handler = handlers.get(step.type)
            if handler:
                inference_steps.extend(handler(step))

        return inference_steps

    def _infer_model_type(self, model_key: str) -> str:
        """
        Infer model type from model key based on predefined patterns.

        Args:
            model_key: The string key of the model.

        Returns:
            String representing the model category (default is "ml").
        """
        key_lower = model_key.lower()
        for model_type, patterns in self.MODEL_PATTERNS.items():
            if any(pattern in key_lower for pattern in patterns):
                return model_type
        return "ml"

    def _extract_inference_info(self, step: Any) -> Dict[str, Any]:
        """
        Extract info from a single inference step.

        Args:
            step: The inference step object.

        Returns:
            Dictionary containing model and wiring metadata.
        """
        inputs = step.wiring.inputs
        return {
            "id": step.id,
            "model_key": inputs.model,
            "features_key": inputs.features,
            "output_key": step.wiring.outputs.predictions,
            "model_type": self._infer_model_type(inputs.model),
        }

    def _extract_branch_inferences(self, branch_step: Any) -> List[Dict[str, Any]]:
        """
        Extract inference info from branch step branches.

        Args:
            branch_step: The step object containing branches.

        Returns:
            List of extracted inferences from valid branches.
        """
        inferences = []
        for branch_name in ["if_true", "if_false"]:
            branch = getattr(branch_step, branch_name, None)
            if branch and getattr(branch, "type", None) == "inference":
                inferences.append(self._extract_inference_info(branch))
        return inferences

    def _extract_sub_pipeline_inferences(
        self, sub_pipeline: Any
    ) -> List[Dict[str, Any]]:
        """
        Extract inference info from sub_pipeline step.

        Args:
            sub_pipeline: The sub-pipeline container step.

        Returns:
            List of inferences found within the sub-pipeline.
        """
        return self._extract_inference_steps(sub_pipeline.pipeline.steps)

    def _extract_load_map(self, steps: List[Any]) -> Dict[str, str]:
        """
        Extract model loading configuration.

        Args:
            steps: List of steps to scan for mlflow_loader.

        Returns:
            Dictionary mapping context keys to step IDs.
        """
        load_map: Dict[str, str] = {}
        for step in steps:
            if step.type == "mlflow_loader":
                for item in step.load_map:
                    load_map[item.context_key] = item.step_id
        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Extract preprocessor configuration.

        Args:
            steps: List of steps to scan.

        Returns:
            Optional dictionary with preprocessor ID and instance key.
        """
        for step in steps:
            if step.type == "preprocessor":
                return {
                    "id": step.id,
                    "instance_key": getattr(step, "instance_key", None),
                }
        return None

    def _get_preprocessor_artifact_name(
        self, preprocessor: Optional[Dict[str, Any]], load_map: Dict[str, str]
    ) -> Optional[str]:
        """
        Get preprocessor artifact name from load_map.

        Args:
            preprocessor: Preprocessor info dict.
            load_map: Loading configuration map.

        Returns:
            The artifact name/ID if found.
        """
        if not preprocessor:
            return None
        return load_map.get(preprocessor.get("instance_key", ""))

    def _extract_data_config(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Extract data and hyperparameter configuration.

        Args:
            cfg: The OmegaConf configuration object.

        Returns:
            Dictionary with data specifications.
        """
        data = getattr(cfg, "data", None)
        hp = getattr(getattr(cfg, "experiment", None), "hyperparams", None)

        return {
            "data_type": getattr(data, "type", "timeseries"),
            "features": list(getattr(data, "features", [])),
            "target_columns": list(getattr(data, "target_columns", [])),
            "input_chunk_length": getattr(hp, "input_chunk_length", 24),
            "output_chunk_length": getattr(hp, "output_chunk_length", 6),
        }
