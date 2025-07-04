from __future__ import annotations

import importlib
import platform
import sys
from importlib.machinery import ModuleSpec

__all__ = ("install_zospy_loader",)


class ZOSPyLoader:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def load_module(fullname):
        if platform.system() != "Windows":
            message = (
                f"Could not import module '{fullname}'.\n"
                "Visisipy's OpticStudio backend depends on ZOSPy, which is not available on non-Windows systems. "
                "To use the OpticStudio backend, use Visisipy on a Windows system."
            )
            raise ImportError(message)

        # If on Windows, attempt to import the actual module
        return importlib.import_module(fullname)


class ZOSPyFinder:
    @staticmethod
    def find_spec(fullname, path, target=None) -> ModuleSpec | None:  # noqa: ARG004
        if fullname.split(".")[0] == "zospy":
            return ModuleSpec(name=fullname, loader=ZOSPyLoader(fullname))

        return None


def install_zospy_loader():
    """Install the ZOSPy loader to handle the import of the ZOSPy module."""
    if "zospy" not in sys.modules and not any(isinstance(m, ZOSPyFinder) for m in sys.meta_path):
        # Only add the finder if it's not already present
        # This prevents multiple finders from being added and causing import errors
        sys.meta_path.insert(0, ZOSPyFinder())
