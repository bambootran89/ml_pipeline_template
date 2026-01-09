from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf


class ConfigGenerator:
    """Expert pipeline configuration automation for eval/serve workloads.

    Loads a base training YAML configuration and generates transformed pipeline
    configs for evaluation or serving workloads. This generator does not modify
    training logic or saved MLflow artifacts, only re-wires dependencies.
    """

    def __init__(self, train_config_path: str) -> None:
        """Initialize ConfigGenerator and load training configuration.

        Args:
            train_config_path: Path to the base training config YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        loaded = OmegaConf.load(train_config_path)
        assert isinstance(
            loaded, DictConfig
        ), f"Expected DictConfig but got {type(loaded).__name__}"
        self.train_cfg: DictConfig = loaded
        self.experiment_name: str = Path(train_config_path).stem

    def _make_evaluator_config(
        self,
        mp: Any,
        init_id: str,
        preprocessor_id: Optional[str],
    ) -> DictConfig:
        """Create evaluator config for a model producer step.

        Args:
            mp: Model producer step (trainer or clustering).
            init_id: MLflow loader step ID.
            preprocessor_id: Optional preprocessor step ID.

        Returns:
            DictConfig for evaluator step.
        """
        base_name = mp.id.replace("_features", "").replace("_model", "")
        eval_id = f"{base_name}_evaluate"

        inputs: Dict[str, Any] = {
            "features": "preprocessed_data",
            "model": f"{mp.id}_model",
        }
        if mp.type != "clustering":
            inputs["targets"] = "target_data"

        step_cfg: Dict[str, Any] = {
            "id": eval_id,
            "type": "evaluator",
            "enabled": True,
            "depends_on": [init_id],
            "wiring": {
                "inputs": inputs,
                "outputs": {"metrics": f"{base_name}_metrics"},
            },
        }

        if mp.type == "clustering":
            step_cfg["step_eval_type"] = "clustering"
        if preprocessor_id:
            step_cfg["depends_on"].append(preprocessor_id)

        return OmegaConf.create(step_cfg)

    def _build_eval_steps(
        self,
        train_steps: List[Any],
        alias: str,
        init_id: str,
        model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for eval pipeline, including evaluator and profiling.

        Args:
            train_steps: Original training pipeline steps.
            alias: MLflow alias for artifact loading.
            init_id: MLflow loader step ID.
            model_producers: Model producers from training steps.
            preprocessor_step: Optional preprocessor step.

        Returns:
            New list of steps for eval mode.
        """
        new_steps: List[Any] = []
        evaluator_ids: List[str] = []

        data_loader = next((s for s in train_steps if s.type == "data_loader"), None)
        if data_loader:
            new_steps.append(data_loader)

        load_map = [
            {"step_id": mp.id, "context_key": f"{mp.id}_model"}
            for mp in model_producers
        ]
        if preprocessor_step:
            load_map.append(
                {
                    "step_id": preprocessor_step.id,
                    "context_key": "transform_manager",
                }
            )

        new_steps.append(
            OmegaConf.create(
                {
                    "id": init_id,
                    "type": "mlflow_loader",
                    "enabled": True,
                    "alias": alias,
                    "load_map": load_map,
                }
            )
        )

        if preprocessor_step:
            prep = copy.deepcopy(preprocessor_step)
            prep.is_train = False
            prep.alias = alias
            # Wiring đúng key restore mà framework serving đang tìm
            prep.instance_key = "transform_manager"
            prep.depends_on = [init_id]
            prep.depends_on.append("load_data")
            new_steps.append(prep)

        for mp in model_producers:
            ev = self._make_evaluator_config(
                mp,
                init_id,
                preprocessor_step.id if preprocessor_step else None,
            )
            new_steps.append(ev)
            evaluator_ids.append(ev.id)

        for step in train_steps:
            if step.type in ["logger", "profiling"]:
                aux = copy.deepcopy(step)
                aux.depends_on = evaluator_ids
                if aux.type == "profiling":
                    aux.exclude_keys = [
                        "cfg",
                        "preprocessor",
                        "df",
                        "train_df",
                        "val_df",
                        "test_df",
                    ]
                new_steps.append(aux)

        return new_steps

    def _build_serve_steps(
        self,
        alias: str,
        init_id: str,
        model_producers: List[Any],
        preprocessor_step: Optional[Any],
    ) -> List[Any]:
        """Build steps for serve pipeline with inference chaining and profiling.

        Ensures preprocessor restore uses correct context key and appears before
        inference chaining begins.

        Args:
            alias: MLflow alias for artifact loading.
            init_id: MLflow loader step ID.
            model_producers: Model producers from training steps.
            preprocessor_step: Optional preprocessor step.

        Returns:
            New list of steps for serve mode.
        """
        new_steps: List[Any] = []
        last_id: str = init_id

        # MLflow loader phải restore cả model và transform_manager
        new_steps.append(
            OmegaConf.create(
                {
                    "id": init_id,
                    "type": "mlflow_loader",
                    "enabled": True,
                    "alias": alias,
                    "load_map": [
                        {"step_id": mp.id, "context_key": f"{mp.id}_model"}
                        for mp in model_producers
                    ]
                    + (
                        [
                            {
                                "step_id": preprocessor_step.id,
                                "context_key": "transform_manager",
                            }
                        ]
                        if preprocessor_step
                        else []
                    ),
                }
            )
        )

        # Preprocessor wiring đúng context key để tránh crash khi serve
        if preprocessor_step:
            prep = copy.deepcopy(preprocessor_step)
            prep.is_train = False
            prep.alias = alias
            prep.instance_key = "transform_manager"
            prep.depends_on = [init_id]
            new_steps.append(prep)
            last_id = prep.id

        # Inference chaining giữ nguyên logic thứ tự và wiring như ban đầu
        for mp in model_producers:
            inf_id = f"{mp.id}_inference"
            inf_step = OmegaConf.create(
                {
                    "id": inf_id,
                    "type": "inference",
                    "enabled": True,
                    "depends_on": [init_id, last_id],
                    "output_as_feature": getattr(mp, "output_as_feature", False),
                    "wiring": {
                        "inputs": {
                            "model": f"{mp.id}_model",
                            "features": "preprocessed_data",
                        },
                        "outputs": {"predictions": f"{mp.id}_predictions"},
                    },
                }
            )
            new_steps.append(inf_step)
            last_id = inf_id

        # Final profiling không đổi logic
        new_steps.append(
            OmegaConf.create(
                {
                    "id": "final_profiling",
                    "type": "profiling",
                    "enabled": True,
                    "depends_on": [last_id],
                    "exclude_keys": [
                        "cfg",
                        "preprocessor",
                        "df",
                        "train_df",
                        "val_df",
                        "test_df",
                    ],
                }
            )
        )

        return new_steps

    def _save_config(self, cfg: DictConfig, path: str) -> None:
        """Save transformed pipeline configuration to YAML file.

        Args:
            cfg: Transformed pipeline DictConfig.
            path: Destination file path.

        Raises:
            OSError: If the file cannot be written.
        """
        with open(path, "w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f)
        print(f"[ConfigGenerator] Successfully generated: {path}")

    def _transform_pipeline(self, mode: str, alias: str) -> DictConfig:
        """Transform base training pipeline into eval or serve pipeline config.

        Args:
            mode: "eval" or "serve".
            alias: MLflow alias for artifact loading.

        Returns:
            Transformed DictConfig for selected mode.

        Raises:
            ValueError: If pipeline.steps is missing.
        """
        cfg = copy.deepcopy(self.train_cfg)

        if "pipeline" not in cfg or "steps" not in cfg.pipeline:
            raise ValueError(
                f"Config {self.experiment_name} is missing pipeline.steps definition"
            )

        train_steps = cfg.pipeline.steps
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
        else:
            new_steps = self._build_serve_steps(
                alias, init_id, model_producers, preprocessor_step
            )

        cfg.pipeline.name = f"{self.experiment_name}_{mode}"
        cfg.pipeline.steps = new_steps

        if "preprocessing" in cfg:
            cfg.preprocessing.is_train = False

        return cfg

    def generate_all(self, output_dir: str, alias: str = "latest") -> Dict[str, str]:
        """Generate both eval and serve pipeline YAML configs and save to files.

        Args:
            output_dir: Directory to store generated configs.
            alias: MLflow alias for artifact loading.

        Returns:
            Dictionary containing saved config paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        eval_path = str(out / f"{self.experiment_name}_eval.yaml")
        serve_path = str(out / f"{self.experiment_name}_serve.yaml")

        cfg_eval = self._transform_pipeline("eval", alias)
        cfg_serve = self._transform_pipeline("serve", alias)

        self._save_config(cfg_eval, eval_path)
        self._save_config(cfg_serve, serve_path)

        return {"eval": eval_path, "serve": serve_path}

    def generate_eval_config(self, alias: str, output_path: str) -> str:
        """Generate evaluation pipeline config and save to file.

        Args:
            alias: MLflow alias.
            output_path: Destination YAML path.

        Returns:
            Path to saved config.
        """
        cfg = self._transform_pipeline("eval", alias)
        self._save_config(cfg, output_path)
        return output_path

    def generate_serve_config(self, alias: str, output_path: str) -> str:
        """Generate serving pipeline config and save to file.

        Args:
            alias: MLflow alias.
            output_path: Destination YAML path.

        Returns:
            Path to saved config.
        """
        cfg = self._transform_pipeline("serve", alias)
        self._save_config(cfg, output_path)
        return output_path
