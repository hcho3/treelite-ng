"""Compatibility layer to enable API migration"""
import ctypes
from typing import Any

from .core import _LIB, _check_call
from .util import c_str


def load_xgboost_model_legacy_binary(filename: str) -> Any:
    """
    Load a tree ensemble model from XGBoost model, stored using
    the legacy binary format.

    TODO(hcho3): Move the implementation to frontend once Model.load()
                 is removed.
    """
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadXGBoostModelEx(
            c_str(filename),
            c_str("{}"),
            ctypes.byref(handle),
        )
    )
    return handle
