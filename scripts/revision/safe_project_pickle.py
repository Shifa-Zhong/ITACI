"""Restricted loader for the project's array-only pickle artifacts.

Only exact NumPy/Pandas constructors observed by the static opcode audit are
permitted. Any other GLOBAL reference raises UnpicklingError before execution.
"""

from __future__ import annotations

import builtins
import importlib
import pickle
from pathlib import Path


ALLOWED_GLOBALS = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.numeric", "_frombuffer"),
    ("pandas", "StringDtype"),
    ("pandas.arrays", "StringArray"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
}

ALLOWED_BUILTINS = {
    "set",
    "frozenset",
    "slice",
    "complex",
    "range",
}


class RestrictedProjectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in ALLOWED_BUILTINS:
            return getattr(builtins, name)
        if (module, name) in ALLOWED_GLOBALS:
            imported = importlib.import_module(module)
            return getattr(imported, name)
        raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")


def safe_load(path_or_file):
    if hasattr(path_or_file, "read"):
        return RestrictedProjectUnpickler(path_or_file).load()
    path = Path(path_or_file)
    with path.open("rb") as handle:
        return RestrictedProjectUnpickler(handle).load()
