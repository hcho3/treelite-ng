"""Treelite Model class"""

from __future__ import annotations

import ctypes
import warnings
from typing import Any, Optional, Union

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
        self.input_type = None
        self.output_type = None
        self.num_tree = None
        self.num_feature = None
        if handle is not None:
            input_type = ctypes.c_char_p()
            output_type = ctypes.c_char_p()
            _check_call(_LIB.TreeliteGetInputType(handle, ctypes.byref(input_type)))
            _check_call(_LIB.TreeliteGetOutputType(handle, ctypes.byref(output_type)))
            self.input_type = py_str(input_type.value)
            self.output_type = py_str(output_type.value)
            num_tree = ctypes.c_size_t()
            num_feature = ctypes.c_int()
            _check_call(_LIB.TreeliteQueryNumTree(handle, ctypes.byref(num_tree)))
            _check_call(_LIB.TreeliteQueryNumFeature(handle, ctypes.byref(num_feature)))
            self.num_tree = num_tree.value
            self.num_feature = num_feature.value

    def __del__(self):
        if self.handle is not None:
            _check_call(_LIB.TreeliteFreeModel(self._handle))
            self._handle = None

    @property
    def handle(self):
        """Access the handle to the associated C++ object"""
        return self._handle

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

    @classmethod
    def from_xgboost(cls, booster: Any) -> Model:
        """
        Deprecated. Please use \ref ~treelite.frontend.from_xgboost instead.
        Load a tree ensemble model from an XGBoost Booster object

        Parameters
        ----------
        booster : Object of type :py:class:`xgboost.Booster`
            Python handle to XGBoost model

        Returns
        -------
        model :
            Loaded model
        """
        return Model(handle=compat.from_xgboost(booster))

    @classmethod
    def from_xgboost_json(
        cls,
        model_json_str: Union[bytes, bytearray, str],
        *,
        allow_unknown_field: bool = False,
    ) -> Model:
        """
        Load a tree ensemble model from a string containing XGBoost JSON

        Parameters
        ----------
        model_json_str :
            A string specifying an XGBoost model in the XGBoost JSON format
        allow_unknown_field:
            Whether to allow extra fields with unrecognized keys

        Returns
        -------
        model
            Loaded model
        """
        return Model(
            handle=compat.from_xgboost_json(
                model_json_str, allow_unknown_field=allow_unknown_field
            )
        )

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
