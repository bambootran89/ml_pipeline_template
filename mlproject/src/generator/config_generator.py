from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from .pipeline.eval_builder import EvalBuilder
from .pipeline.serve_builder import ServeBuilder
from .pipeline.tune_builder import TuneBuilder


class ConfigGenerator:
    """Generate pipeline configs for eval/serve/tune from training config.

    Loads a base training YAML configuration and generates transformed
    pipeline configs for evaluation or serving workloads. Does not modify
    training logic or saved MLflow artifacts.
    """

    def __init__(self, train_config_path: str) -> None:
        """Initialize config generator.

        Args:
            train_config_path: Path to training config YAML file.
        """
        loaded = OmegaConf.load(train_config_path)

        if not isinstance(loaded, DictConfig):
            raise TypeError(f"Expected DictConfig but got {type(loaded).__name__}")

        self.train_cfg: DictConfig = loaded
        self.experiment_name: str = Path(train_config_path).stem

        train_steps = self.train_cfg.pipeline.steps

        self._eval_builder = EvalBuilder(train_steps)
        self._serve_builder = ServeBuilder(train_steps)
        self._tune_builder = TuneBuilder(train_steps)

    def generate_all(
        self,
        output_dir: str,
        alias: str = "latest",
        include_tune: bool = False,
    ) -> Dict[str, str]:
        """Generate eval, serve, and optionally tune configs.

        Args:
            output_dir: Directory to save generated configs.
            alias: Model alias for loading artifacts.
            include_tune: Whether to generate tune config.

        Returns:
            Dict mapping mode name to output file path.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        eval_file = str(out_path / f"{self.experiment_name}_eval.yaml")
        eval_cfg = self._transform_pipeline("eval", alias)
        self._save_config(eval_cfg, eval_file)
        paths["eval"] = eval_file

        serve_file = str(out_path / f"{self.experiment_name}_serve.yaml")
        serve_cfg = self._transform_pipeline("serve", alias)
        self._save_config(serve_cfg, serve_file)
        paths["serve"] = serve_file

        if include_tune:
            tune_file = str(out_path / f"{self.experiment_name}_tune.yaml")
            tune_cfg = self._transform_pipeline("tune")
            self._save_config(tune_cfg, tune_file)
            paths["tune"] = tune_file

        return paths

    def generate_eval_config(self, alias: str, output_path: str) -> str:
        """Generate evaluation pipeline config.

        Args:
            alias: Model alias for loading artifacts.
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config.

        Args:
            alias: Model alias for loading artifacts.
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_tune_config(self, output_path: str) -> str:
        """Generate tuning pipeline config.

        Args:
            output_path: Path to save generated config.

        Returns:
            Path to generated config file.
        """
        cfg = self._transform_pipeline("tune")
        self._save_config(cfg, output_path)
        return output_path

    def _transform_pipeline(self, mode: str, alias: str = "latest") -> DictConfig:
        """Transform training pipeline to target mode.

        Args:
            mode: Target mode - 'eval', 'serve', or 'tune'.
            alias: Model alias for loading artifacts.

        Returns:
            Transformed pipeline config.

        Raises:
            ValueError: If pipeline config is invalid or mode unknown.
        """
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(f"Config {self.experiment_name} missing pipeline.steps")

        train_steps = cfg.pipeline.steps

        if mode == "tune":
            new_steps = self._tune_builder.build()
        else:
            preprocessor_step = self._find_preprocessor(train_steps)
            init_id = "init_artifacts"

            if mode == "eval":
                new_steps = self._eval_builder.build(alias, init_id, preprocessor_step)
            elif mode == "serve":
                new_steps = self._serve_builder.build(alias, init_id, preprocessor_step)
            else:
                raise ValueError(
                    f"Unknown mode: {mode}. Must be 'eval', 'serve', or 'tune'"
                )

        cfg.pipeline.name = f"{self.experiment_name}_{mode}"
        cfg.pipeline.steps = new_steps

        if "preprocessing" in cfg and mode != "tune":
            cfg.preprocessing.is_train = False

        return cfg

    def _find_preprocessor(self, steps: list) -> Optional[Any]:
        """Find legacy preprocessor step.

        Args:
            steps: List of pipeline steps.

        Returns:
            Preprocessor step if found, None otherwise.
        """
        return next((s for s in steps if s.type == "preprocessor"), None)

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save pipeline config to YAML file.

        Args:
            cfg: Config to save.
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as file:
            OmegaConf.save(config=cfg, f=file)

        print(f"[ConfigGenerator] Successfully generated: {path}")
