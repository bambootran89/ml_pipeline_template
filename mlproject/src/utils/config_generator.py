"""Config generator for eval/serve pipelines from training config.

This module auto-generates evaluation and serving pipeline configurations
from training experiment YAML files.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from mlproject.src.utils.config_class import ConfigLoader


class ConfigGenerator:
    """Generate eval/serve configs from training config.

    This class transforms a training experiment configuration into
    evaluation and serving configurations by:
    - Adjusting preprocessing to load from MLflow (is_train=False)
    - Adding model_loader step instead of trainer
    - Removing tuning-related steps
    - Preserving experiment metadata for reproducibility

    Attributes
    ----------
    train_cfg : DictConfig
        Original training configuration.
    experiment_name : str
        Name of the experiment.
    """

    EVAL_PIPELINE_TEMPLATE = {
        "pipeline": {
            "name": "auto_generated_eval",
            "steps": [
                {"id": "load_data", "type": "data_loader", "enabled": True},
                {
                    "id": "preprocess",
                    "type": "preprocessor",
                    "enabled": True,
                    "depends_on": ["load_data"],
                    "is_train": False,
                    "alias": "latest",
                },
                {
                    "id": "load_model",
                    "type": "model_loader",
                    "enabled": True,
                    "depends_on": ["preprocess"],
                    "alias": "latest",
                },
                {
                    "id": "evaluate",
                    "type": "evaluator",
                    "enabled": True,
                    "depends_on": ["load_model", "preprocess"],
                    "model_step_id": "load_model",
                },
                {
                    "id": "profiling",
                    "type": "profiling",
                    "enabled": True,
                    "depends_on": ["evaluate"],
                },
            ],
        }
    }

    SERVE_PIPELINE_TEMPLATE = {
        "pipeline": {
            "name": "auto_generated_serve",
            "steps": [
                {
                    "id": "preprocess",
                    "type": "preprocessor",
                    "enabled": True,
                    "is_train": False,
                    "alias": "latest",
                },
                {
                    "id": "load_model",
                    "type": "model_loader",
                    "enabled": True,
                    "depends_on": ["preprocess"],
                    "alias": "latest",
                },
                {
                    "id": "inference",
                    "type": "inference",
                    "enabled": True,
                    "depends_on": ["preprocess", "load_model"],
                    "model_step_id": "load_model",
                },
            ],
        }
    }

    def __init__(self, train_config_path: str) -> None:
        """Initialize config generator.

        Parameters
        ----------
        train_config_path : str
            Path to training experiment YAML file.

        Raises
        ------
        FileNotFoundError
            If config file does not exist.
        """
        path = Path(train_config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {train_config_path}")

        self.train_cfg = ConfigLoader.load(train_config_path)
        self.experiment_name = self.train_cfg.get("experiment", {}).get("name", "")
        self.source_path = train_config_path

    def generate_eval_config(
        self,
        alias: str = "latest",
        output_path: Optional[str] = None,
    ) -> DictConfig:
        """Generate evaluation pipeline config.

        This module auto-generates evaluation and serving pipeline configurations
        from training experiment YAML files.

        The pipeline is created by:
        1. Setting preprocessing to inference mode (is_train=False)
        2. Injecting model_loader in place of training steps
        3. Removing tuning and training-only keys
        4. Propagating MLflow alias for model/preprocessor pairing
        5. Preserving experiment metadata for reproducibility
        """
        eval_cfg = self._copy_base_config()

        pipeline: Dict[str, Any] = copy.deepcopy(
            self.EVAL_PIPELINE_TEMPLATE["pipeline"]
        )
        pipeline["name"] = f"{self.experiment_name}_eval"

        # --- FIX: ensure writable list typing ---
        raw_steps: Any = pipeline.get("steps", [])
        steps: List[Dict[str, Any]] = list(raw_steps)  # type: ignore

        for step in steps:
            if step.get("alias") is not None:
                step["alias"] = alias

        pipeline["steps"] = steps
        eval_cfg["pipeline"] = pipeline

        eval_cfg["_generated"] = {
            "source": str(self.source_path),
            "type": "eval",
            "alias": alias,
        }

        if output_path:
            self._save_config(eval_cfg, output_path)

        return eval_cfg

    def generate_serve_config(
        self,
        alias: str = "latest",
        output_path: Optional[str] = None,
    ) -> DictConfig:
        """Generate serving pipeline config.

        This module auto-generates evaluation and serving pipeline configurations
        from training experiment YAML files.

        The serve pipeline supports:
        - Feast offline feature materialization
        - MLflow model/preprocessor alias pairing
        - Online inference deployment payload generation
        - Minimal pipeline mutation for maximum compatibility
        """
        serve_cfg = self._copy_base_config()

        pipeline: Dict[str, Any] = copy.deepcopy(
            self.SERVE_PIPELINE_TEMPLATE["pipeline"]
        )
        pipeline["name"] = f"{self.experiment_name}_serve"

        # --- FIX: ensure writable list typing ---
        raw_steps: Any = pipeline.get("steps", [])
        steps: List[Dict[str, Any]] = list(raw_steps)  # type: ignore

        for step in steps:
            if step.get("alias") is not None:
                step["alias"] = alias

        pipeline["steps"] = steps
        serve_cfg["pipeline"] = pipeline

        serve_cfg["_generated"] = {
            "source": str(self.source_path),
            "type": "serve",
            "alias": alias,
        }

        if output_path:
            self._save_config(serve_cfg, output_path)

        return serve_cfg

    def generate_all(
        self,
        output_dir: str,
        alias: str = "latest",
    ) -> Dict[str, str]:
        """Generate both eval and serve configs.

        Parameters
        ----------
        output_dir : str
            Directory to save generated configs.
        alias : str, default="latest"
            MLflow model alias.

        Returns
        -------
        Dict[str, str]
            Paths to generated configs {"eval": path, "serve": path}.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(self.source_path).stem

        eval_path = str(out_dir / f"{base_name}_eval.yaml")
        serve_path = str(out_dir / f"{base_name}_serve.yaml")

        self.generate_eval_config(alias=alias, output_path=eval_path)
        self.generate_serve_config(alias=alias, output_path=serve_path)

        return {"eval": eval_path, "serve": serve_path}

    def _copy_base_config(self) -> DictConfig:
        """Create a deep copy of base config without pipeline.

        Returns
        -------
        DictConfig
            Copied config without pipeline key.
        """
        container = OmegaConf.to_container(self.train_cfg, resolve=True)
        if not isinstance(container, dict):
            container = {}

        # Remove training-specific keys
        keys_to_remove = ["pipeline", "tuning", "_generated"]
        for key in keys_to_remove:
            container.pop(key, None)

        return OmegaConf.create(container)

    def _save_config(self, cfg: DictConfig, output_path: str) -> None:
        """Save config to YAML file.

        Parameters
        ----------
        cfg : DictConfig
            Configuration to save.
        output_path : str
            Output file path.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            OmegaConf.save(cfg, f)

        print(f"[ConfigGenerator] Saved: {output_path}")


def generate_configs_cli() -> None:
    """CLI entry point for config generation."""
    parser = argparse.ArgumentParser(
        description="Generate eval/serve configs from training config"
    )
    parser.add_argument(
        "--train-config",
        "-t",
        required=True,
        help="Path to training experiment YAML",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="mlproject/configs/generated",
        help="Output directory for generated configs",
    )
    parser.add_argument(
        "--alias",
        "-a",
        default="latest",
        help="MLflow model alias (latest/production/staging)",
    )
    parser.add_argument(
        "--type",
        choices=["eval", "serve", "all"],
        default="all",
        help="Type of config to generate",
    )

    args = parser.parse_args()

    generator = ConfigGenerator(args.train_config)

    if args.type == "all":
        paths = generator.generate_all(args.output_dir, args.alias)
        print("\nGenerated configs:")
        print(f"  - Eval:  {paths['eval']}")
        print(f"  - Serve: {paths['serve']}")
    elif args.type == "eval":
        out_path = str(
            Path(args.output_dir) / f"{Path(args.train_config).stem}_eval.yaml"
        )
        generator.generate_eval_config(args.alias, out_path)
    else:
        out_path = str(
            Path(args.output_dir) / f"{Path(args.train_config).stem}_serve.yaml"
        )
        generator.generate_serve_config(args.alias, out_path)


if __name__ == "__main__":
    generate_configs_cli()
