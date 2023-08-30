"""Treelite Model class"""

from __future__ import annotations

import ctypes
from typing import Any, Optional

from .core import _LIB, _check_call
from .util import c_str, py_str


class Model:
    """
    Decision tree ensemble model

    Parameters
    ----------
    handle :
        Initial value of model handle
    """

    def __init__(self, *, handle: Optional[Any] = None):
        self._handle = handle

    @classmethod
    def load_xgboost_model(cls, filename: str) -> Model:
        """
        Load a tree ensemble model from XGBoost model

        Parameters
        ----------
        filename :
            Path to model file

        Returns
        -------
        model :
            loaded model
        """
        handle = ctypes.c_void_p()
        _check_call(
            _LIB.TreeliteLoadXGBoostModelEx(
                c_str(filename),
                c_str("{}"),
                ctypes.byref(handle),
            )
        )
        return Model(handle=handle)

    def dump_as_json(self, *, pretty_print: bool = True) -> str:
        """
        Dump the model as a JSON string. This is useful for inspecting details of the tree
        ensemble model.

        Parameters
        ----------
        pretty_print :
            Whether to pretty-print the JSON string, set this to False to make the string compact

        Returns
        -------
        json_str :
            JSON string representing the model
        """
        json_str = ctypes.c_char_p()
        _check_call(
            _LIB.TreeliteDumpAsJSON(
                self._handle,
                ctypes.c_int(1 if pretty_print else 0),
                ctypes.byref(json_str),
            )
        )
        return py_str(json_str.value)
