"""Model loader ingest scikit-learn models into Treelite"""

from .importer import import_model


def import_model_with_model_builder(sklearn_model):
    """
    Load a tree ensemble model from a scikit-learn model object using the model builder API.

    .. note:: Use ``import_model`` for production use

        This function exists to demonstrate the use of the model builder API and is slow with
        large models. For production, please use :py:func:`~treelite.sklearn.import_model`
        which is significantly faster.
    """
    raise NotImplementedError("Not implemented")


__all__ = ["import_model", "import_model_with_model_builder"]
