from __future__ import annotations

from visisipy import analysis, models, opticstudio, plots, refraction, wavefront
from visisipy.backend import get_backend, set_backend
from visisipy.models import (
    EyeGeometry,
    EyeMaterials,
    EyeModel,
    NavarroGeometry,
    NavarroMaterials,
)

__all__ = (
    "models",
    "EyeModel",
    "EyeGeometry",
    "EyeMaterials",
    "NavarroGeometry",
    "NavarroMaterials",
    "analysis",
    "opticstudio",
    "get_backend",
    "set_backend",
    "plots",
    "refraction",
    "wavefront",
)

__version__ = "0.0.1"