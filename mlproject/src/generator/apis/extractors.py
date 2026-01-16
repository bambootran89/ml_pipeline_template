"""Mixins for extracting API logic from pipeline configurations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ApiGeneratorExtractorsMixin:
    """Extractions logic for API generation.

    Splits large configuration pipelines into minimal inference steps
    by identifying model-loading nodes and their dependencies.
    """

    # Model type patterns for type inference
    MODEL_PATTERNS = {
        "dl": ["darts", "pytorch", "tensorflow", "keras", "lstm", "tft", "nhits"],
        "ml": ["sklearn", "xgboost", "catboost", "lightgbm", "regression", "forest"],
    }

    def _extract_inference_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Extract all model inference steps from pipeline.

        This identifies steps that perform model prediction and collects
        their configuration for code generation.

        Args:
            steps: List of pipeline steps.

        Returns:
            List of extracted inference information.
        """
        inference_steps: List[Dict[str, Any]] = []

        # Dispatcher map for step handlers
        handlers = {
            "training": self._get_training_inference,
            "clustering": self._get_training_inference,
            "inference": self._get_inference_info,
            "feature_inference": self._get_inference_info,
        }

        for step in steps:
            step_type = getattr(step, "type", None) or step.get("type")
            handler = handlers.get(step_type)
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

    def _get_training_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Extract inference info from a training step."""
        step_id = getattr(step, "id", None) or step.get("id")
        inputs = getattr(step, "inputs", []) or step.get("inputs", [])

        model_key = step_id
        model_type = self._infer_model_type(model_key)

        return [
            {
                "id": f"{step_id}_inference",
                "model_key": model_key,
                "model_type": model_type,
                "inputs": list(inputs),
                "features_key": "preprocessed_data",
                "output_key": f"{step_id}_predictions",
            }
        ]

    def _get_inference_info(self, step: Any) -> List[Dict[str, Any]]:
        """Extract info from an explicit inference step."""
        step_id = getattr(step, "id", None) or step.get("id")
        inputs = getattr(step, "inputs", []) or step.get("inputs", [])

        # Use get for args if it exists, otherwise check for model_key/source_model_key in step directly
        model_key = None
        args = getattr(step, "args", None) or step.get("args")
        if args:
            model_key = args.get("model_key")

        if model_key is None:
            model_key = (
                step.get("source_model_key")
                or step.get("model_key")
                or step_id.replace("_inference", "")
            )

        model_type = self._infer_model_type(model_key)

        # Handle different field names for features/output
        features_key = (
            step.get("features_key")
            or step.get("base_features_key")
            or "preprocessed_data"
        )
        output_key = step.get("output_key") or f"{model_key}_predictions"

        return [
            {
                "id": step_id,
                "model_key": model_key,
                "model_type": model_type,
                "inputs": list(inputs),
                "features_key": features_key,
                "output_key": output_key,
            }
        ]

    def _extract_load_map(self, steps: List[Any]) -> Dict[str, str]:
        """Extract mapping of model keys to MLflow artifact names."""
        load_map = {}
        for step in steps:
            step_type = getattr(step, "type", None) or step.get("type")
            if step_type in ["init_artifacts", "mlflow_loader"]:
                # Handle artifacts from init_artifacts or mlflow_loader step
                items = (
                    step.get("load_map")
                    or step.get("artifacts")
                    or step.get("args", {}).get("artifacts", {})
                )

                if items:
                    # In OmegaConf, iterate over ListConfig or DictConfig works
                    # No need to check isinstance(items, list) strictly as ListConfig is iterable
                    for item in items:
                        # Check if it has get (DictConfig and dict have it)
                        if hasattr(item, "get"):
                            key = item.get("context_key") or item.get("step_id")
                            val = item.get("step_id")
                            if key and val:
                                load_map[key] = val
                        elif isinstance(items, dict):
                            # items itself is a dict (old format)
                            for k, v in items.items():
                                if any(
                                    x in k.lower()
                                    for x in ["model", "preprocess", "scaler"]
                                ):
                                    load_map[k] = v
                            break  # Exiting for loop after processing dict

        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract preprocessor configuration if present."""
        preprocessor_step = next(
            (
                s
                for s in steps
                if (getattr(s, "type", None) or s.get("type")) == "preprocessor"
            ),
            None,
        )
        if preprocessor_step:
            return {
                "id": getattr(preprocessor_step, "id", None)
                or preprocessor_step.get("id"),
                "instance_key": preprocessor_step.get("instance_key"),
                "args": getattr(preprocessor_step, "args", None) or preprocessor_step,
            }

        # Check for preprocessor in init_artifacts / mlflow_loader
        init_step = next(
            (
                s
                for s in steps
                if (getattr(s, "type", None) or s.get("type"))
                in ["init_artifacts", "mlflow_loader"]
            ),
            None,
        )
        if init_step:
            items = (
                init_step.get("load_map")
                or init_step.get("artifacts")
                or init_step.get("args", {}).get("artifacts", {})
            )
            if items:
                for item in items:
                    if hasattr(item, "get"):
                        key = item.get("context_key") or item.get("step_id")
                        val = item.get("step_id")
                        if key and (
                            "preprocess" in key.lower() or "scaler" in key.lower()
                        ):
                            return {"id": key, "artifact_name": val}
                    elif isinstance(items, dict):
                        for k, v in items.items():
                            if "preprocess" in k.lower() or "scaler" in k.lower():
                                return {"id": k, "artifact_name": v}
                        break

        return None

    def _get_preprocessor_artifact_name(
        self, preprocessor: Optional[Dict[str, Any]], load_map: Dict[str, str]
    ) -> Optional[str]:
        """Extract preprocessor artifact name from configuration or load map."""
        if not preprocessor:
            return None

        # Try artifact_name directly
        if "artifact_name" in preprocessor:
            return preprocessor["artifact_name"]

        # Try instance_key first (as it's often the key in load_map)
        prep_key = preprocessor.get("instance_key")
        if prep_key and prep_key in load_map:
            return load_map[prep_key]

        # Try to find in load_map using id
        prep_id = preprocessor.get("id")
        if prep_id and prep_id in load_map:
            return load_map[prep_id]

        # Fallback to id if it looks like a preprocessor
        if prep_id and ("preprocess" in prep_id.lower() or "scaler" in prep_id.lower()):
            return prep_id

        return None

    def _extract_data_config(self, cfg: Any) -> Dict[str, Any]:
        """Extract data configuration (type, features, windowing)."""
        data_cfg = {}

        # 1. Try to get from data section
        data = getattr(cfg, "data", None) or cfg.get("data")
        if data:
            data_cfg["data_type"] = data.get("type", "timeseries")
            data_cfg["features"] = list(data.get("features", []))

        # 2. Try to get from experiment section (overrides data)
        experiment = getattr(cfg, "experiment", None) or cfg.get("experiment")
        if experiment:
            data_cfg["data_type"] = experiment.get("type", data_cfg.get("data_type"))

        # 3. Look for timeseries specific windowing
        # Search through all steps for datamodule args
        pipeline = getattr(cfg, "pipeline", None) or cfg.get("pipeline")
        if pipeline and "steps" in pipeline:
            for step in pipeline.steps:
                dm_args = None
                args = getattr(step, "args", None) or step.get("args")
                if args and "datamodule" in args:
                    dm_args = args.datamodule
                elif "datamodule" in step:
                    dm_args = step.datamodule

                if dm_args:
                    if "input_chunk_length" in dm_args:
                        data_cfg["input_chunk_length"] = dm_args.input_chunk_length
                    if "output_chunk_length" in dm_args:
                        data_cfg["output_chunk_length"] = dm_args.output_chunk_length

        # Defaults
        data_cfg.setdefault("data_type", "timeseries")
        data_cfg.setdefault("features", [])
        data_cfg.setdefault("input_chunk_length", 24)
        data_cfg.setdefault("output_chunk_length", 6)

        return data_cfg
