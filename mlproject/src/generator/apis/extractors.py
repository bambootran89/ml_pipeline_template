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
            "branch": self._get_branch_inference,
            "sub_pipeline": self._get_nested_inference,
            "parallel": self._get_nested_inference,
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
            # Check wiring for model input if not explicitly set
            wiring = step.get("wiring")
            if wiring:
                inputs = wiring.get("inputs")
                if inputs:
                    model_key = inputs.get("model")

        if model_key is None:
            model_key = (
                step.get("source_model_key")
                or step.get("model_key")
                or step_id.replace("_inference", "")
            )

        model_type = self._infer_model_type(model_key)

        # Handle different field names for features/output
        features_key = step.get("features_key") or step.get("base_features_key")

        if not features_key:
            wiring = step.get("wiring")
            if wiring and wiring.get("inputs"):
                inputs = wiring.get("inputs")
                # heuristic to find input data key
                for key in ["features", "X", "data", "input"]:
                    if key in inputs:
                        features_key = inputs[key]
                        break

        if not features_key:
            features_key = "preprocessed_data"

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

    def _get_branch_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Recursively extract inference info from branch steps."""
        results = []
        if_true = step.get("if_true")
        if_false = step.get("if_false")

        if if_true:
            results.extend(self._extract_inference_steps([if_true]))
        if if_false:
            results.extend(self._extract_inference_steps([if_false]))
        return results

    def _get_nested_inference(self, step: Any) -> List[Dict[str, Any]]:
        """Recursively extract inference info from sub_pipeline or parallel steps."""
        pipeline = step.get("pipeline")
        if pipeline and "steps" in pipeline:
            return self._extract_inference_steps(pipeline.steps)

        # Handle parallel directly if it's a list of steps
        steps = step.get("steps")
        if steps:
            return self._extract_inference_steps(steps)

        return []

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
                    for item in items:
                        if hasattr(item, "get"):
                            key = item.get("context_key") or item.get("step_id")
                            val = item.get("step_id")
                            if key and val:
                                load_map[key] = val
                        elif isinstance(items, dict):
                            for k, v in items.items():
                                if any(
                                    x in k.lower()
                                    for x in ["model", "preprocess", "scaler"]
                                ):
                                    load_map[k] = v
                            break

            # Recursively explore nested steps for load_map (unlikely but possible)
            elif step_type == "branch":
                load_map.update(
                    self._extract_load_map([step.get("if_true"), step.get("if_false")])
                )
            elif step_type in ["sub_pipeline", "parallel"]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    load_map.update(self._extract_load_map(nested_steps))

        return load_map

    def _extract_preprocessor_info(self, steps: List[Any]) -> Optional[Dict[str, Any]]:
        """Extract preprocessor configuration if present."""
        for step in steps:
            if not step:
                continue
            step_type = getattr(step, "type", None) or step.get("type")

            if step_type == "preprocessor":
                return {
                    "id": getattr(step, "id", None) or step.get("id"),
                    "instance_key": step.get("instance_key"),
                    "args": getattr(step, "args", None) or step,
                }

            if step_type in ["init_artifacts", "mlflow_loader"]:
                items = (
                    step.get("load_map")
                    or step.get("artifacts")
                    or step.get("args", {}).get("artifacts", {})
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

            # Recursive search
            if step_type == "branch":
                info = self._extract_preprocessor_info(
                    [step.get("if_true"), step.get("if_false")]
                )
                if info:
                    return info
            elif step_type in ["sub_pipeline", "parallel"]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    info = self._extract_preprocessor_info(nested_steps)
                    if info:
                        return info

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
            exp_type = experiment.get("type")
            if exp_type in ["classification", "regression"]:
                data_cfg["data_type"] = "tabular"
            elif exp_type in ["multivariate", "univariate"]:
                data_cfg["data_type"] = "timeseries"
            elif exp_type:
                # Use experiment type if data type not yet set
                if "data_type" not in data_cfg:
                    data_cfg["data_type"] = exp_type

        # 3. Look for timeseries specific windowing
        # Search through all steps for datamodule args
        pipeline = getattr(cfg, "pipeline", None) or cfg.get("pipeline")
        if pipeline and "steps" in pipeline:
            data_cfg.update(self._extract_datamodule_recursive(pipeline.steps))

        # Defaults
        data_cfg.setdefault("data_type", "timeseries")
        data_cfg.setdefault("features", [])
        data_cfg.setdefault("input_chunk_length", 24)
        data_cfg.setdefault("output_chunk_length", 6)

        return data_cfg

    def _extract_datamodule_recursive(self, steps: List[Any]) -> Dict[str, Any]:
        """Recursively search for datamodule configuration in steps."""
        data_cfg = {}
        for step in steps:
            if not step:
                continue
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
                return data_cfg  # Found, stop searching

            # Recursive search
            step_type = getattr(step, "type", None) or step.get("type")
            if step_type == "branch":
                res = self._extract_datamodule_recursive(
                    [step.get("if_true"), step.get("if_false")]
                )
                if res:
                    return res
            elif step_type in ["sub_pipeline", "parallel"]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    res = self._extract_datamodule_recursive(nested_steps)
                    if res:
                        return res
        return data_cfg
