import os
from typing import Optional, Union, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


def _ensure_dict(cfg: Union[DictConfig, ListConfig]) -> DictConfig:
    """
    Ensure that a given OmegaConf object is a DictConfig.

    Args:
        cfg (Union[DictConfig, ListConfig]): Input configuration object.

    Returns:
        DictConfig: The input converted to a DictConfig if it was a ListConfig.
    """
    if isinstance(cfg, DictConfig):
        return cfg

    # OmegaConf.create(ListConfig) always produces a DictConfig
    created = OmegaConf.create(cfg)  # type: ignore[arg-type]
    return cast(DictConfig, created)


def _load_default_file(base_dir: str, entry: Union[str, dict]) -> DictConfig:
    """
    Load a default configuration file relative to a base directory.

    Args:
        base_dir (str): Base directory to resolve file paths.
        entry (Union[str, dict]):
        Either a string filename or a dict mapping key to stub.

    Returns:
        DictConfig: Loaded configuration as a DictConfig.
        Returns empty DictConfig if file not found.
    """
    if isinstance(entry, str):
        candidate = os.path.join(base_dir, entry)
        if os.path.exists(candidate):
            return _ensure_dict(OmegaConf.load(candidate))
        return OmegaConf.create({})

    _, file_stub = next(iter(entry.items()))
    candidate = os.path.join(base_dir, f"{file_stub}.yaml")
    if os.path.exists(candidate):
        return _ensure_dict(OmegaConf.load(candidate))
    return OmegaConf.create({})


def load_config(cfg_path: str = "") -> DictConfig:
    """
    Load an experiment configuration file with support for merging defaults.

    Args:
        cfg_path (str, optional):
        Path to the main configuration YAML file.
        Defaults to 'mlproject/configs/experiments/etth1.yaml'.

    Returns:
        DictConfig: The merged configuration including defaults and main config.
    """
    if not cfg_path:
        cfg_path = os.path.join("mlproject", "configs", "experiments", "etth1.yaml")

    cfg: DictConfig = OmegaConf.load(cfg_path)  # type: ignore[assignment]

    # Get the defaults list if defined
    defaults: Optional[ListConfig] = cfg.get("defaults")  # type: ignore[assignment]
    if defaults is None:
        return cfg

    base_dir = os.path.dirname(cfg_path)
    merged: DictConfig = OmegaConf.create({})

    # Merge default configurations
    for entry in defaults:
        sub = _load_default_file(base_dir, entry)
        merged = OmegaConf.merge(merged, sub)  # type: ignore[assignment]

    # Merge main config last to allow overrides
    merged = OmegaConf.merge(merged, cfg)  # type: ignore[assignment]
    return merged
