from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

from omegaconf import DictConfig, OmegaConf

from .apis.mixin import ApiGeneratorMixin
from .pipeline.eval_mixin import EvalPipelineMixin
from .pipeline.serve_mixin import ServePipelineMixin
from .pipeline.tune_mixin import TuneMixin


# pylint: disable=too-many-ancestors
class ConfigGenerator(
    EvalPipelineMixin, ServePipelineMixin, TuneMixin, ApiGeneratorMixin
):
    """Expert pipeline configuration automation for eval/serve workloads.

    Loads a base training YAML configuration and generates transformed pipeline
    configs for evaluation or serving workloads. This generator does not modify
    training logic or saved MLflow artifacts, only re-wires dependencies.

    Also supports generating API code (FastAPI or Ray Serve) from serve configs.
    """

    def __init__(self, train_config_path: str) -> None:
        """Initialize ConfigGenerator and load training configuration."""
        loaded = OmegaConf.load(train_config_path)
        assert isinstance(
            loaded, DictConfig
        ), f"Expected DictConfig but got {type(loaded).__name__}"
        self.train_cfg: DictConfig = loaded
        self.experiment_name: str = Path(train_config_path).stem

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save transformed pipeline configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        print(f"[ConfigGenerator] Successfully generated: {path}")

    def _transform_pipeline(self, mode: str, alias: str = "latest") -> DictConfig:
        """Transform base training pipeline into eval, serve, or tune config."""
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(f"Config {self.experiment_name} missing pipeline.steps")

        train_steps = cfg.pipeline.steps

        if mode == "tune":
            new_steps = self._build_tune_steps(train_steps)
        else:
            model_producers = [
                s for s in train_steps if s.type in ["trainer", "clustering"]
            ]
            preprocessor_step = next(
                (s for s in train_steps if s.type == "preprocessor"), None
            )
            init_id = "init_artifacts"

            if mode == "eval":
                new_steps = self._build_eval_steps(
                    train_steps, alias, init_id, model_producers, preprocessor_step
                )
            elif mode == "serve":
                train_steps = self.train_cfg.pipeline.steps
                new_steps = self._build_serve_steps(
                    alias, init_id, model_producers, preprocessor_step, train_steps
                )
            else:
                raise ValueError(
                    f"Unknown mode: {mode}. Must be 'eval', 'serve', or 'tune'"
                )

        cfg.pipeline.name = f"{self.experiment_name}_{mode}"
        cfg.pipeline.steps = new_steps

        if "preprocessing" in cfg and mode != "tune":
            cfg.preprocessing.is_train = False

        return cfg

    def generate_all(
        self, output_dir: str, alias: str = "latest", include_tune: bool = False
    ) -> Dict[str, str]:
        """Generate eval and serve (optionally tune) pipeline YAML configs."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}

        eval_path = str(out / f"{self.experiment_name}_eval.yaml")
        cfg_eval = self._transform_pipeline("eval", alias)
        self._save_config(cfg_eval, eval_path)
        paths["eval"] = eval_path

        serve_path = str(out / f"{self.experiment_name}_serve.yaml")
        cfg_serve = self._transform_pipeline("serve", alias)
        self._save_config(cfg_serve, serve_path)
        paths["serve"] = serve_path

        if include_tune:
            tune_path = str(out / f"{self.experiment_name}_tune.yaml")
            cfg_tune = self._transform_pipeline("tune")
            self._save_config(cfg_tune, tune_path)
            paths["tune"] = tune_path

        return paths

    def generate_eval_config(self, alias: str, output_path: str) -> str:
        """Generate evaluation pipeline config and save to file."""
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config and save to file."""
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_tune_config(self, output_path: str) -> str:
        """Generate tuning pipeline config and save to file."""
        cfg = self._transform_pipeline("tune")
        self._save_config(cfg, output_path)
        return output_path
