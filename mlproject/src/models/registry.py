from importlib import import_module
from typing import Dict


def get_model_class(entry: Dict[str, str]):
    """
    entry: {module: "...", class: "..."}
    """
    modname = entry["module"]
    clsname = entry["class"]
    mod = import_module(modname)
    return getattr(mod, clsname)
