from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig


class ApiGeneratorExtractorsMixin:
    """Extraction helpers for API generator."""

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Extract inference steps from pipeline."""
        inference_steps = []
        for step in steps:
            if step.type == "inference":
                inference_steps.append(self._extract_inference_info(step))
            elif step.type == "branch":
                inference_steps.extend(self._extract_branch_inferences(step))
            elif step.type == "sub_pipeline":
                inference_steps.extend(self._extract_sub_pipeline_inferences(step))
        return inference_steps

    def _infer_model_type(self, model_key: str) -> str:
        """Infer model type from model key."""
        key_lower = model_key.lower()
        ml_patterns = [
            "xgboost",
            "xgb",
            "catboost",
            "kmeans",
            "kmean",
            "lightgbm",
            "lgbm",
            "randomforest",
            "rf",
        ]
        for pattern in ml_patterns:
            if pattern in key_lower:
                return "ml"

        dl_patterns = ["tft", "nlinear", "transformer", "lstm", "gru", "rnn"]
        for pattern in dl_patterns:
            if pattern in key_lower:
                return "deep_learning"

        return "ml"

    def _extract_inference_info(self, step: Any) -> Dict[str, Any]:
        """Extract info from single inference step."""
        model_key = step.wiring.inputs.model
        return {
            "id": step.id,
            "model_key": model_key,
            "features_key": step.wiring.inputs.features,
            "output_key": step.wiring.outputs.predictions,
            "model_type": self._infer_model_type(model_key),
        }

    def _extract_branch_inferences(self, branch_step: Any) -> List[Dict[str, Any]]:
        """Extract inference info from branch step."""
        inferences = []
        for branch_name in ["if_true", "if_false"]:
            if hasattr(branch_step, branch_name):
                branch = getattr(branch_step, branch_name)
                if branch.type == "inference":
                    inferences.append(self._extract_inference_info(branch))
        return inferences

    def _extract_sub_pipeline_inferences(
        self, sub_pipeline: Any
    ) -> List[Dict[str, Any]]:
        """Extract inference info from sub_pipeline step."""
        inferences = []
        print(sub_pipeline)
        for step in sub_pipeline.pipeline.steps:
            if step.type == "inference":
                inferences.append(self._extract_inference_info(step))
        return inferences

    def _extract_load_map(self, steps: List[Any]) -> Dict[str, str]:
        """Extract model loading configuration."""
        load_map = {}
        for step in steps:
            if step.type == "mlflow_loader":
                for item in step.load_map:
                    load_map[item.context_key] = item.step_id
        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract preprocessor configuration."""
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
        """Get preprocessor artifact name from load_map."""
        if not preprocessor:
            return None
        instance_key = preprocessor.get("instance_key")
        if not instance_key:
            return None
        return load_map.get(instance_key)

    def _extract_data_config(self, cfg: DictConfig) -> Dict[str, Any]:
        """Extract data configuration from config."""
        data_config = {
            "data_type": "timeseries",
            "features": [],
            "target_columns": [],
            "input_chunk_length": 24,
            "output_chunk_length": 6,
        }

        if hasattr(cfg, "data"):
            data = cfg.data
            data_config["data_type"] = getattr(data, "type", "timeseries")
            data_config["features"] = list(getattr(data, "features", []))
            data_config["target_columns"] = list(getattr(data, "target_columns", []))

        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "hyperparams"):
            hp = cfg.experiment.hyperparams
            data_config["input_chunk_length"] = getattr(hp, "input_chunk_length", 24)
            data_config["output_chunk_length"] = getattr(hp, "output_chunk_length", 6)

        return data_config
