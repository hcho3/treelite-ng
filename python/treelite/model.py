"""Treelite Model class"""

from __future__ import annotations

from typing import Any, Optional

from .core import _LIB


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
        return Model(handle=_LIB.load_xgboost_model(filename, "{}"))

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
        return _LIB.dump_as_json(self._handle, pretty_print)
