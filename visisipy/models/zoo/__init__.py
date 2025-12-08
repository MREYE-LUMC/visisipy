"""Predefined schematic eyes."""

from __future__ import annotations

from visisipy.models.zoo.bennett_rabbetts import BennettRabbettsEyeModel, BennettRabbettsGeometry
from visisipy.models.zoo.gullstrand import GullstrandLeGrandEyeModel, GullstrandLeGrandGeometry
from visisipy.models.zoo.navarro import NavarroGeometry

__all__ = (
    "BennettRabbettsEyeModel",
    "BennettRabbettsGeometry",
    "GullstrandLeGrandEyeModel",
    "GullstrandLeGrandGeometry",
    "NavarroGeometry",
)
