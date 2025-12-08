"""Optical eye models.

This module provides the base classes for eye models and surfaces.
"""

from __future__ import annotations

from visisipy.models import materials
from visisipy.models.base import BaseEye, BaseSurface, EyeModel
from visisipy.models.factory import create_geometry
from visisipy.models.geometry import EyeGeometry
from visisipy.models.materials import (
    EyeMaterials,
    NavarroMaterials,
    NavarroMaterials458,
    NavarroMaterials543,
    NavarroMaterials589,
    NavarroMaterials633,
)
from visisipy.models.zoo import (
    BennettRabbettsEyeModel,
    BennettRabbettsGeometry,
    GullstrandLeGrandEyeModel,
    GullstrandLeGrandGeometry,
    NavarroGeometry,
)

__all__ = (
    "BaseEye",
    "BaseSurface",
    "BennettRabbettsEyeModel",
    "BennettRabbettsGeometry",
    "EyeGeometry",
    "EyeMaterials",
    "EyeModel",
    "GullstrandLeGrandEyeModel",
    "GullstrandLeGrandGeometry",
    "NavarroGeometry",
    "NavarroMaterials",
    "NavarroMaterials458",
    "NavarroMaterials543",
    "NavarroMaterials589",
    "NavarroMaterials633",
    "create_geometry",
    "materials",
)
