import os
from typing import Any, Dict, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigLoader:
    """
    Class-based configuration loader using OmegaConf.

    Features:
    ---------
    - Load experiment YAML
    - Resolve and merge Hydra-style default config blocks
    - Convert ListConfig → DictConfig safely
    - Return a single merged DictConfig without mypy errors

    Example:
    --------
    defaults:
      - data: base/data
      - model: base/model
      - preprocessing: base/preprocessing
      - training: base/training
      - evaluation: base/evaluation
    """

    @staticmethod
    def _ensure_dict(cfg: Union[DictConfig, ListConfig]) -> DictConfig:
        """
        Ensure that an OmegaConf object is returned as DictConfig.
        """
        if isinstance(cfg, DictConfig):
            return cfg

        # ListConfig → DictConfig conversion
        created = OmegaConf.create(cfg)
        return cast(DictConfig, created)

    @staticmethod
    def _load_default_file(
        base_dir: str, entry: Union[str, Dict[str, Any]]
    ) -> DictConfig:
        """
        Load a default YAML config block listed under `defaults`.
        """
        # Case 1: entry: "data.yaml"
        if isinstance(entry, str):
            candidate = os.path.join(base_dir, entry)
            if os.path.exists(candidate):
                return ConfigLoader._ensure_dict(OmegaConf.load(candidate))
            return OmegaConf.create({})

        # Case 2: entry: {"data": "base/data"}
        _, file_stub = next(iter(entry.items()))
        candidate = os.path.join(base_dir, f"{file_stub}.yaml")
        if os.path.exists(candidate):
            return ConfigLoader._ensure_dict(OmegaConf.load(candidate))

        return OmegaConf.create({})

    @staticmethod
    def load(cfg_path: str = "") -> DictConfig:
        """
        Load main experiment config, merge with default config files.

        Args:
            cfg_path:
                Path to experiment YAML.
                Defaults to:
                mlproject/configs/experiments/etth1.yaml

        Returns:
            DictConfig: A fully merged OmegaConf object.
        """
        # Default experiment file
        if not cfg_path:
            cfg_path = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")

        # Load primary config
        cfg_raw = OmegaConf.load(cfg_path)
        cfg: DictConfig = cast(DictConfig, cfg_raw)

        # If no defaults: return raw config directly
        defaults = cfg.get("defaults")  # type: ignore[assignment]
        if defaults is None:
            return cfg

        base_dir = os.path.dirname(cfg_path)
        merged: DictConfig = OmegaConf.create({})

        # Merge default blocks
        for entry in cast(ListConfig, defaults):
            sub_cfg = ConfigLoader._load_default_file(base_dir, entry)
            merged = cast(DictConfig, OmegaConf.merge(merged, sub_cfg))

        # Merge main config last (highest priority)
        final_cfg = cast(DictConfig, OmegaConf.merge(merged, cfg))

        return final_cfg
