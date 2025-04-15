from __future__ import annotations

from visisipy import analysis, models, opticstudio, optiland, plots, refraction, wavefront
from visisipy.backend import get_backend, set_backend
from visisipy.models import (
    EyeGeometry,
    EyeMaterials,
    EyeModel,
    NavarroGeometry,
    NavarroMaterials,
)

__all__ = (
    "EyeGeometry",
    "EyeMaterials",
    "EyeModel",
    "NavarroGeometry",
    "NavarroMaterials",
    "analysis",
    "get_backend",
    "models",
    "opticstudio",
    "optiland",
    "plots",
    "refraction",
    "set_backend",
    "wavefront",
)

__version__ = "0.0.1"
