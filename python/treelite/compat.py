"""Compatibility layer to enable API migration"""
import ctypes
import json
from typing import Any

from .core import _LIB, _check_call
from .util import c_str


def load_xgboost_model_legacy_binary(filename: str) -> Any:
    """
    Load a tree ensemble model from XGBoost model, stored using
    the legacy binary format.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.load() is removed.
    """
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadXGBoostModelLegacyBinary(
            c_str(filename), c_str("{}"), ctypes.byref(handle)
        )
    )
    return handle


def load_xgboost_model(filename: str, *, allow_unknown_field: bool) -> Any:
    """
    Load a tree ensemble model from XGBoost model, stored using the JSON format.

    TODO(hcho3): Move the implementation to treelite.frontend once
                 Model.load() is removed.
    """
    parser_config = {"allow_unknown_field": allow_unknown_field}
    parser_config_str = json.dumps(parser_config)
    handle = ctypes.c_void_p()
    _check_call(
        _LIB.TreeliteLoadXGBoostModel(
            c_str(filename), c_str(parser_config_str), ctypes.byref(handle)
        )
    )
    return handle
