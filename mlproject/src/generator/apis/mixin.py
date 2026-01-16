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

        if framework == "fastapi":
            code = self._generate_fastapi_code(
                pipeline_name,
                self._extract_load_map(steps),
                self._extract_preprocessor_info(steps),
                self._extract_inference_steps(steps),
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
                self._extract_inference_steps(steps),
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
