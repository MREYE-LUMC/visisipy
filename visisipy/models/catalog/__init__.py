"""Predefined schematic eyes."""

from __future__ import annotations

from visisipy.models.catalog.bennett_rabbetts import BennettRabbettsEyeModel, BennettRabbettsGeometry
from visisipy.models.catalog.gullstrand import GullstrandLeGrandEyeModel, GullstrandLeGrandGeometry
from visisipy.models.catalog.navarro import NavarroGeometry

__all__ = (
    "BennettRabbettsEyeModel",
    "BennettRabbettsGeometry",
    "GullstrandLeGrandEyeModel",
    "GullstrandLeGrandGeometry",
    "NavarroGeometry",
)
