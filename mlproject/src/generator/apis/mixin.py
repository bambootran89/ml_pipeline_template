"""API code generation mixin for FastAPI and Ray Serve.

This mixin generates API code from serve pipeline configurations,
supporting both FastAPI and Ray Serve frameworks.

Supports:
- Tabular data: batch prediction for multiple rows
- Timeseries data: multi-step prediction with configurable horizon
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .fastapi import ApiGeneratorFastAPIMixin
from .ray import ApiGeneratorRayServeMixin


class ApiGeneratorMixin(ApiGeneratorFastAPIMixin, ApiGeneratorRayServeMixin):
    """Mixin for generating API code from serve configurations.

    Provides methods to generate FastAPI and Ray Serve code from
    serve pipeline YAML configs. Uses template-based generation
    to keep complexity low.

    Supports both tabular and timeseries data types with appropriate
    prediction strategies (batch for tabular, multi-step for timeseries).
    """

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
        alias: str = "production",
    ) -> str:
        """Generate API code from serve configuration."""
        cfg = OmegaConf.load(serve_config_path)
        assert isinstance(cfg, DictConfig)

        pipeline_name = cfg.pipeline.name
        steps = cfg.pipeline.steps

        # Extract data config from experiment config (preferred) or serve config
        if experiment_config_path and Path(experiment_config_path).exists():
            exp_cfg = OmegaConf.load(experiment_config_path)
            assert isinstance(exp_cfg, DictConfig)
            data_config = self._extract_data_config(exp_cfg)
        else:
            data_config = self._extract_data_config(cfg)

        # IMPORTANT: Extract feature generators from SERVE config (has pipeline structure)
        # This is separate from data_config extraction because serve config has
        # the transformed sub-pipeline steps that define feature generators
        feature_generators = self._extract_feature_generators(list(steps))
        if feature_generators:
            data_config["feature_generators"] = feature_generators
            print(f"[ApiGenerator] Found {len(feature_generators)} feature generators")
            for fg in feature_generators:
                print(f"  - {fg['step_id']} -> {fg['output_key']}")

        # Extract inference steps, excluding feature generators
        inference_steps = self._extract_inference_steps_excluding_generators(
            steps, feature_generators
        )

        if framework == "fastapi":
            code = self._generate_fastapi_code(
                pipeline_name,
                self._extract_load_map(steps),
                self._extract_preprocessor_info(steps),
                inference_steps,
                experiment_config_path,
                data_config,
                alias=alias,
            )
            filename = f"{pipeline_name}_fastapi.py"
        elif framework == "ray":
            code = self._generate_ray_serve_code(
                pipeline_name,
                self._extract_load_map(steps),
                self._extract_preprocessor_info(steps),
                inference_steps,
                experiment_config_path,
                data_config,
                alias=alias,
            )
            filename = f"{pipeline_name}_ray.py"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        print(f"[ApiGenerator] Generated {framework} API: {output_path}")
        print(f"[ApiGenerator] Data type: {data_config.get('data_type', 'unknown')}")
        return str(output_path)

    def _extract_inference_steps_excluding_generators(self, steps, feature_generators):
        """Extract inference steps, excluding feature generator steps."""
        # Get all inference steps
        all_inference = self._extract_inference_steps(steps)

        # Get feature generator step IDs to exclude
        fg_step_ids = {fg["step_id"] for fg in feature_generators}
        fg_model_keys = {fg["model_key"] for fg in feature_generators}

        # Filter out feature generators from inference steps
        filtered = []
        for inf in all_inference:
            model_key = inf.get("model_key", "")
            step_id = inf.get("id", "").replace("_inference", "")

            # Skip if this is a feature generator
            if step_id in fg_step_ids or model_key in fg_model_keys:
                print(
                    f"[ApiGenerator] Excluding feature generator from inference: {step_id}"
                )
                continue

            # Also skip by model_key pattern
            if f"fitted_{step_id}" in fg_model_keys:
                print(
                    f"[ApiGenerator] Excluding feature generator from inference: {step_id}"
                )
                continue

            filtered.append(inf)

        return filtered
