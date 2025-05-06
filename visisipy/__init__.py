"""Visisipy: VISion Simulations In PYthon."""

from __future__ import annotations

import platform
from importlib.metadata import version

from visisipy import analysis, models, optiland, plots, refraction, wavefront
from visisipy._zospy_loader import install_zospy_loader
from visisipy.backend import get_backend, set_backend, update_settings
from visisipy.models import (
    EyeGeometry,
    EyeMaterials,
    EyeModel,
    NavarroGeometry,
    NavarroMaterials,
    create_geometry,
)

__all__ = [
    "EyeGeometry",
    "EyeMaterials",
    "EyeModel",
    "NavarroGeometry",
    "NavarroMaterials",
    "analysis",
    "create_geometry",
    "get_backend",
    "models",
    "optiland",
    "plots",
    "refraction",
    "set_backend",
    "update_settings",
    "wavefront",
]

# The OpticStudio backend is only available on Windows
if platform.system() == "Windows":
    from visisipy import opticstudio

    __all__ += ["opticstudio"]

__version__ = version("visisipy")

install_zospy_loader()
