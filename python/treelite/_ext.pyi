from typing import Any

class TreeliteError(RuntimeError): ...

__version__: str

def load_xgboost_model(filename: str, config: str) -> Any: ...
def dump_as_json(model: Any, pretty_print: bool) -> str: ...