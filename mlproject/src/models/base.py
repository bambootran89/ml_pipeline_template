import json
import os
from abc import ABC, abstractmethod
from typing import Optional, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


class ModelWrapperBase(ABC):
    """
    Base class for all model wrappers.

    Responsibilities:
        - Build model architecture
        - Predict (in child)
        - Save / Load model weights and metadata
        - Ensure model is initialized
    """

    def __init__(self, cfg: Optional[dict] = None):
        """
        Initialize the wrapper with a configuration.

        Args:
            cfg (dict or DictConfig, optional): configuration dictionary or DictConfig
        """
        if cfg is None:
            self.cfg = DictConfig({})
        elif isinstance(cfg, DictConfig):
            self.cfg = cfg
        elif isinstance(cfg, dict):
            self.cfg = OmegaConf.create(cfg)
        else:
            raise TypeError("cfg must be dict or DictConfig")

        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    @abstractmethod
    def build(self, input_dim: int, output_dim: int):
        """Build model architecture. Must be implemented by subclasses."""
        raise NotImplementedError

    def ensure_built(self):
        """Raise an error if model is not yet built."""
        if self.model is None:
            raise RuntimeError(
                "Model is not built yet. Call build(input_dim, output_dim)."
            )

    def save(self, save_dir: str):
        """
        Save model weights and metadata.

        Args:
            save_dir (str): directory path to save model.pt and meta.json
        """
        self.ensure_built()
        os.makedirs(save_dir, exist_ok=True)

        assert self.model is not None
        # Save weights
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))

        # Ensure cfg is DictConfig
        cfg_to_save = self.cfg
        if not isinstance(cfg_to_save, DictConfig):
            cfg_to_save = OmegaConf.create(cfg_to_save)

        # Save metadata
        meta = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "cfg": OmegaConf.to_container(cfg_to_save, resolve=True),
        }

        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, save_dir: str):
        """
        Load model weights and rebuild from metadata.

        Args:
            save_dir (str): directory path containing model.pt and meta.json

        Returns:
            self
        """
        meta_path = os.path.join(save_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise RuntimeError("meta.json not found — model cannot be rebuilt")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Ensure cfg is DictConfig
        if not isinstance(self.cfg, DictConfig):
            self.cfg = DictConfig({})

        # Merge metadata cfg
        loaded_cfg = OmegaConf.create(meta["cfg"])
        self.cfg = cast(DictConfig, OmegaConf.merge(self.cfg, loaded_cfg))

        # Save dims
        self.cfg.input_dim = meta["input_dim"]
        self.cfg.output_dim = meta["output_dim"]

        # Rebuild model
        self.build(meta["input_dim"], meta["output_dim"])

        # Load weights
        weight_path = os.path.join(save_dir, "model.pt")
        assert self.model is not None
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))

        print(f"[Model Loaded] {weight_path}")
        return self
