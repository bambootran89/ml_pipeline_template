from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from .apis.mixin import ApiGeneratorMixin


class ApiGenerator:
    """Expert pipeline configuration automation for eval/serve workloads.

    Loads a base training YAML configuration and generates transformed pipeline
    configs for evaluation or serving workloads. This generator does not modify
    training logic or saved MLflow artifacts, only re-wires dependencies.

    Also supports generating API code (FastAPI or Ray Serve) from serve configs.
    """

    def __init__(
        self,
    ) -> None:
        """Initialize ConfigGenerator and load training configuration."""
        self._api_generator = ApiGeneratorMixin()

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save transformed pipeline configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        print(f"[ConfigGenerator] Successfully generated: {path}")

    def generate_api(
        self,
        serve_config_path: str,
        output_dir: str,
        framework: str = "fastapi",
        experiment_config_path: str = "",
        alias: str = "production",
    ) -> str:
        """Generate API code from serve configuration.

        Delegates to ApiGeneratorMixin.
        """
        return self._api_generator.generate_api(
            serve_config_path,
            output_dir,
            framework,
            experiment_config_path,
            alias=alias,
        )
