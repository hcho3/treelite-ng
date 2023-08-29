"""Interface with pybind11 compiled extension"""
import warnings

# pylint: disable=import-error, no-name-in-module, consider-using-from-import
import treelite._ext as _LIB


def _log_callback(msg: str) -> None:
    """Redirect logs from native library into Python console"""
    print(msg)


def _warn_callback(msg: str) -> None:
    """Redirect warnings from native library into Python console"""
    warnings.warn(msg)


_LIB.register_callback_log_info(_log_callback)
_LIB.register_callback_log_warning(_warn_callback)

__all__ = ["_LIB"]
