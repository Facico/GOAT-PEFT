"""
MoLE Initialization
"""
from .config import GOATConfig
from .layer import GOATLayer, LinearGOATLayer
from .model import GOATModel

__all__ = ["GOATConfig", "GOATLayer", "LinearGOATLayer", "GOATModel"]


def __getattr__(name):
    raise AttributeError(f"Module {__name__} has no attribute {name}.")
