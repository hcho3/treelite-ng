"""Treelite Model class"""

from __future__ import annotations

import ctypes
import warnings
from typing import Any, Optional

from . import compat
from .core import _LIB, _check_call
from .util import py_str


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
    def load(
        cls, filename: str, model_format: str, allow_unknown_field: bool = False
    ) -> Model:
        """
        Deprecated. Please use \ref ~treelite.frontend.load_xgboost_model instead.
        Load a tree ensemble model from a file

        Parameters
        ----------
        filename :
            Path to model file
        model_format :
            Model file format. Must be "xgboost", "xgboost_json", or "lightgbm"
        allow_unknown_field:
            Whether to allow extra fields with unrecognized keys. This flag is only
            applicable if model_format="xgboost_json"

        Returns
        -------
        model :
            Loaded model
        """
        model_format = model_format.lower()

        def deprecation_warning(alt: str) -> None:
            warnings.warn(
                (
                    "treelite.Model.load() is deprecated. "
                    f"Use treelite.frontend.{alt}() instead."
                ),
                FutureWarning,
            )

        if model_format == "xgboost":
            deprecation_warning("load_xgboost_model_legacy_binary")
            return Model(handle=compat.load_xgboost_model_legacy_binary(filename))
        if model_format == "xgboost_json":
            deprecation_warning("load_xgboost_model")
            return Model(
                handle=compat.load_xgboost_model(
                    filename, allow_unknown_field=allow_unknown_field
                )
            )
        raise NotImplementedError("Not implemented yet")

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
