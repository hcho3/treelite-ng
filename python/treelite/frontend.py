"""Functions to load and build model objects"""
from __future__ import annotations

from . import compat
from .model import Model


def load_xgboost_model_legacy_binary(filename: str) -> Model:
    """
    Load a tree ensemble model from XGBoost model, stored using
    the legacy binary format. Note: new XGBoost models should
    be stored in the JSON format, to take advantage of the
    latest functionalities of XGBoost.

    Parameters
    ----------
    filename :
        Path to model file

    Returns
    -------
    model :
        Loaded model

    Example
    -------

    .. code-block:: python

       xgb_model = treelite.frontend.load_xgboost_model_legacy_binary(
           "xgboost_model.model")
    """
    return Model(handle=compat.load_xgboost_model_legacy_binary(filename))


__all__ = ["load_xgboost_model_legacy_binary"]
