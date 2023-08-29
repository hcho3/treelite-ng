"""Treelite module"""
from .core import _LIB
from .model import Model

__version__ = _LIB.__version__

__all__ = ["Model", "__version__"]
