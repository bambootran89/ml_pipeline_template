"""Mixins for extracting API logic from pipeline configurations."""
# flake8: noqa: C901
# pylint: disable=R1702, R0912

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

    def _sort_by_deps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort steps by additional_feature_keys dependencies."""
        if not steps:
            return steps

        # Map output_key -> step
        out_map = {s.get("output_key"): s for s in steps if s.get("output_key")}

        # Build deps: step_id -> [dep_ids]
        deps: Dict[str, List[str]] = {}
        by_id = {}
        for s in steps:
            sid = s.get("id", "")
            by_id[sid] = s
            deps[sid] = []
            additional_keys = s.get("additional_feature_keys") or []
            for k in additional_keys:
                if k in out_map:
                    dep = out_map[k].get("id", "")
                    if dep and dep != sid:
                        deps[sid].append(dep)

        # Topological sort
        result: List[Dict[str, Any]] = []
        visited: set = set()
        visiting: set = set()

        def visit(sid: str) -> None:
            if sid in visited:
                return
            if sid in visiting:
                return  # Cycle, skip
            visiting.add(sid)
            for dep in deps.get(sid, []):
                if dep in by_id:
                    visit(dep)
            visiting.remove(sid)
            visited.add(sid)
            if sid in by_id:
                result.append(by_id[sid])

        for sid in deps:
            if sid not in visited:
                visit(sid)

        # Add remaining
        for s in steps:
            if s not in result:
                result.append(s)

        return result

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

        # Use get for args if it exists, otherwise check for
        # model_key/source_model_key in step directly
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

        # Extract additional_feature_keys if present
        additional_feature_keys = step.get("additional_feature_keys")
        if additional_feature_keys:
            additional_feature_keys = list(additional_feature_keys)

        return [
            {
                "id": step_id,
                "model_key": model_key,
                "model_type": model_type,
                "inputs": list(inputs),
                "features_key": features_key,
                "output_key": output_key,
                "additional_feature_keys": additional_feature_keys,
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
        data_cfg: Dict[str, Any] = {}

        # 1. Try to get from data section
        data = getattr(cfg, "data", None) or cfg.get("data")
        if data:
            data_cfg["data_type"] = data.get("type", "timeseries")
            data_cfg["features"] = list(data.get("features", []))
            data_cfg["path"] = data.get("path", "")
            data_cfg["entity_key"] = data.get("entity_key", "")

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

        # 3. Look for timeseries specific windowing and additional_feature_keys
        # Search through all steps for datamodule args
        pipeline = getattr(cfg, "pipeline", None) or cfg.get("pipeline")
        if pipeline and "steps" in pipeline:
            data_cfg.update(self._extract_datamodule_recursive(pipeline.steps))
            # Extract feature generators from sub-pipelines
            feature_generators = self._extract_feature_generators(pipeline.steps)
            if feature_generators:
                data_cfg["feature_generators"] = feature_generators

        # Defaults
        data_cfg.setdefault("data_type", "timeseries")
        data_cfg.setdefault("features", [])
        data_cfg.setdefault("input_chunk_length", 24)
        data_cfg.setdefault("output_chunk_length", 6)
        data_cfg.setdefault("additional_feature_keys", [])
        data_cfg.setdefault("feature_generators", [])

        return data_cfg

    def _extract_feature_generators(
        self, steps: List[Any], parent_pipeline_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract feature generator steps based on additional_feature_keys.

        This method finds steps that OUTPUT the keys listed in additional_feature_keys
        from the datamodule step. This is more reliable than detecting by step type.

        Args:
            steps: List of pipeline steps.
            parent_pipeline_id: ID of parent sub_pipeline if any.

        Returns:
            List of feature generator configurations.
        """
        # Step 1: Find additional_feature_keys from datamodule or
        # feature_inference steps
        _ = parent_pipeline_id
        additional_keys = self._find_additional_feature_keys(steps)
        if not additional_keys:
            print("[ApiGenerator] No additional_feature_keys found in config")
            return []

        print(f"[ApiGenerator] Looking for steps that output: {additional_keys}")

        # Step 2: Build a map of output_key -> step for all steps (recursive)
        output_map = self._build_output_key_map(steps)

        # Step 3: Find steps that output the additional_feature_keys
        generators: List[Dict[str, Any]] = []
        for key in additional_keys:
            if key in output_map:
                step = output_map[key]
                fg_info = self._build_feature_generator_info_from_key(step, key)
                if fg_info:
                    generators.append(fg_info)
                    print(f"  Found: {fg_info['step_id']} -> {key}")
            else:
                print(f"  Warning: No step outputs '{key}'")

        return generators

    def _find_additional_feature_keys(self, steps: List[Any]) -> List[str]:
        """Find additional_feature_keys from datamodule or feature_inference steps."""
        keys: List[str] = []

        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Check datamodule step
            if step_type == "datamodule":
                additional = step.get("additional_feature_keys")
                if additional:
                    keys.extend(list(additional))

            # Check feature_inference step
            if step_type == "feature_inference":
                additional = step.get("additional_feature_keys")
                if additional:
                    keys.extend(list(additional))

            # Recursively check sub-pipelines
            if step_type == "sub_pipeline":
                pipeline = step.get("pipeline")
                if pipeline and "steps" in pipeline:
                    keys.extend(
                        self._find_additional_feature_keys(list(pipeline.steps))
                    )

        return list(set(keys))  # Remove duplicates

    def _build_output_key_map(self, steps: List[Any]) -> Dict[str, Any]:
        """Build a map of output_key -> step for all steps (recursive)."""
        output_map: Dict[str, Any] = {}

        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Extract output keys from wiring
            wiring = step.get("wiring")
            if wiring and wiring.get("outputs"):
                outputs = wiring.outputs
                if hasattr(outputs, "items"):
                    for _, context_key in outputs.items():
                        output_map[str(context_key)] = step

            # Recursively process sub-pipelines
            if step_type == "sub_pipeline":
                pipeline = step.get("pipeline")
                if pipeline and "steps" in pipeline:
                    nested_map = self._build_output_key_map(list(pipeline.steps))
                    output_map.update(nested_map)

            elif step_type == "parallel":
                nested_steps = step.get("steps") or []
                nested_map = self._build_output_key_map(nested_steps)
                output_map.update(nested_map)

        return output_map

    def _build_feature_generator_info_from_key(
        self, step: Any, output_key: str
    ) -> Optional[Dict[str, Any]]:
        """Build feature generator info from step and known output key."""
        step_id = getattr(step, "id", None) or step.get("id")
        step_type = getattr(step, "type", None) or step.get("type")

        if not step_id:
            return None

        # Determine inference method based on step type
        inference_method = "transform"
        if step_type in ["clustering", "trainer"]:
            inference_method = "predict"
        elif step_type == "inference":
            # Check step_id for hints
            if "cluster" in step_id.lower():
                inference_method = "predict"
        elif step.get("run_method"):
            run_method = str(step.run_method)
            if "predict" in run_method:
                inference_method = "predict"

        # # Determine fg_type from step_id or step_type
        # fg_type = "transform"
        # step_id_lower = step_id.lower()
        # if step_type == "clustering" or "cluster" in step_id_lower:
        #     fg_type = "clustering"
        # elif "pca" in step_id_lower:
        #     fg_type = "pca"

        return {
            "step_id": step_id,
            "model_key": f"fitted_{step_id}",
            "artifact_name": step_id,
            "output_key": output_key,
            "inference_method": inference_method,
            "step_type": step_type,
        }

    def _is_feature_generator(self, step: Any) -> bool:
        """Check if step is a feature generator.

        A step is a feature generator if it:
        - Has output_as_feature=True
        - Is a clustering step with output key
        - Is a dynamic_adapter with artifact_type=preprocess and output_as_feature
        - Has wiring.outputs that produces feature-like keys

        Args:
            step: Pipeline step to check.

        Returns:
            True if step generates features for downstream use.
        """
        if not step:
            return False

        if step.get("output_as_feature", False):
            return True

        step_type = getattr(step, "type", None) or step.get("type")

        if step_type == "clustering":
            return self._check_clustering_generator(step)

        if step_type == "dynamic_adapter":
            return self._check_dynamic_adapter_generator(step)

        if step_type == "inference":
            return self._check_inference_generator(step)

        return False

    def _check_clustering_generator(self, step: Any) -> bool:
        """Check if clustering step is a feature generator."""
        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Has feature output (not just model)
            for key in ["features", "cluster_labels", "predictions"]:
                if key in outputs:
                    return True
        return True  # Clustering typically generates features

    def _check_dynamic_adapter_generator(self, step: Any) -> bool:
        """Check if dynamic adapter step is a feature generator."""
        class_path = step.get("class_path", "").lower()

        # EXCLUDE preprocessors that modify columns in place
        if any(x in class_path for x in ["imputer", "scaler", "normalizer", "encoder"]):
            return False

        # PCA, clustering add NEW columns - these are feature generators
        if any(x in class_path for x in ["pca", "cluster", "kmeans"]):
            return True

        # Check output_as_feature flag for other dynamic adapters
        if step.get("output_as_feature", False):
            return True
        return False

    def _check_inference_generator(self, step: Any) -> bool:
        """Check if inference step is a feature generator."""
        step_id = getattr(step, "id", None) or step.get("id", "")
        step_id_lower = step_id.lower()

        # EXCLUDE preprocessors (impute, scale) - they don't add new columns
        if any(x in step_id_lower for x in ["impute", "scale", "normalize"]):
            return False

        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Check if it outputs features (not just predictions)
            if "features" in outputs:
                return True

        # Only cluster and pca add new columns
        if any(x in step_id_lower for x in ["cluster", "pca"]):
            return True
        return False

    def _build_feature_generator_info(self, step: Any) -> Optional[Dict[str, Any]]:
        """Build feature generator info from step.

        Args:
            step: Feature generator step.

        Returns:
            Feature generator configuration dict or None.
        """
        step_id = getattr(step, "id", None) or step.get("id")
        step_type = getattr(step, "type", None) or step.get("type")

        if not step_id:
            return None

        # Determine output key
        output_key = self._determine_fg_output_key(step, step_id)

        # Determine inference method
        inference_method = self._determine_fg_inference_method(step, step_type, step_id)

        # Determine step type for categorization
        fg_type = self._determine_fg_type(step, step_type, step_id)

        return {
            "step_id": step_id,
            "model_key": f"fitted_{step_id}",
            "artifact_name": step_id,
            "output_key": output_key,
            "inference_method": inference_method,
            "step_type": fg_type,
        }

    def _determine_fg_output_key(self, step: Any, step_id: str) -> str:
        """Determine output key for feature generator."""
        output_key = None
        wiring = step.get("wiring")
        if wiring and wiring.get("outputs"):
            outputs = wiring.outputs
            # Try common output keys
            for key in ["features", "cluster_labels", "predictions", "data"]:
                if key in outputs:
                    output_key = outputs[key]
                    break
            # Fallback to first output
            if not output_key and hasattr(outputs, "keys"):
                keys = list(outputs.keys())
                if keys:
                    output_key = outputs[keys[0]]

        if not output_key:
            output_key = f"{step_id}_features"
        return output_key

    def _determine_fg_inference_method(
        self, step: Any, step_type: str, step_id: str
    ) -> str:
        """Determine inference method for feature generator."""
        inference_method = "transform"
        if step_type == "clustering":
            inference_method = "predict"
        elif step_type == "inference":
            # For transformed inference steps, check step_id for type hints
            step_id_lower = step_id.lower()
            if "cluster" in step_id_lower:
                inference_method = "predict"
            else:
                inference_method = "transform"
        elif step.get("run_method"):
            run_method = step.run_method
            if "transform" in run_method:
                inference_method = "transform"
            elif "predict" in run_method:
                inference_method = "predict"
        return inference_method

    def _determine_fg_type(self, step: Any, step_type: str, step_id: str) -> str:
        """Determine feature generator type."""
        fg_type = "transform"
        if step_type == "clustering":
            fg_type = "clustering"
        elif step_type == "inference":
            # For transformed inference steps, infer type from step_id
            step_id_lower = step_id.lower()
            if "cluster" in step_id_lower or "kmeans" in step_id_lower:
                fg_type = "clustering"
            elif "pca" in step_id_lower:
                fg_type = "pca"
            elif "impute" in step_id_lower:
                fg_type = "imputer"
            elif "scaler" in step_id_lower:
                fg_type = "scaler"
        elif step_type == "dynamic_adapter":
            class_path = step.get("class_path", "").lower()
            if "pca" in class_path:
                fg_type = "pca"
            elif "imputer" in class_path or "impute" in class_path:
                fg_type = "imputer"
            elif "scaler" in class_path:
                fg_type = "scaler"
            elif "cluster" in class_path or "kmeans" in class_path:
                fg_type = "clustering"
        return fg_type

    def _extract_datamodule_recursive(self, steps: List[Any]) -> Dict[str, Any]:
        """Recursively search for datamodule configuration in steps."""
        data_cfg: Dict[str, Any] = {}
        for step in steps:
            if not step:
                continue

            step_type = getattr(step, "type", None) or step.get("type")

            # Check for datamodule step with additional_feature_keys
            if step_type == "datamodule":
                additional_keys = step.get("additional_feature_keys")
                if additional_keys:
                    data_cfg["additional_feature_keys"] = list(additional_keys)

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
                # Don't return early if we found dm_args but need to continue searching
                # for additional_feature_keys

            # Recursive search
            if step_type == "branch":
                res = self._extract_datamodule_recursive(
                    [step.get("if_true"), step.get("if_false")]
                )
                if res:
                    data_cfg.update(res)
            elif step_type in ["sub_pipeline", "parallel"]:
                nested_steps = step.get("steps") or (
                    step.get("pipeline", {}).get("steps")
                )
                if nested_steps:
                    res = self._extract_datamodule_recursive(nested_steps)
                    if res:
                        data_cfg.update(res)

        return data_cfg
