"""Analyze optical eye models in Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.backend import BaseAnalysisRegistry, _AnalysisMethod
from visisipy.optiland.analysis.cardinal_points import cardinal_points
from visisipy.optiland.analysis.raytrace import raytrace
from visisipy.optiland.analysis.refraction import refraction
from visisipy.optiland.analysis.zernike_coefficients import zernike_standard_coefficients

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend

__all__ = ("OptilandAnalysisRegistry",)


class OptilandAnalysisRegistry(BaseAnalysisRegistry):
    """Analyses for the OpticStudio backend."""

    def __init__(self, backend: OptilandBackend):
        super().__init__(backend)
        self._optic = backend.optic

    cardinal_points = _AnalysisMethod(cardinal_points)
    raytrace = _AnalysisMethod(raytrace)
    zernike_standard_coefficients = _AnalysisMethod(zernike_standard_coefficients)
    refraction = _AnalysisMethod(refraction)
