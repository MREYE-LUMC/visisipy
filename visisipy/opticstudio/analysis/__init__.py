from typing import TYPE_CHECKING

from visisipy.backend import BaseAnalysis, _AnalysisMethod
from visisipy.opticstudio.analysis.cardinal_points import cardinal_points
from visisipy.opticstudio.analysis.raytrace import raytrace
from visisipy.opticstudio.analysis.refraction import refraction
from visisipy.opticstudio.analysis.zernike_coefficients import (
    zernike_standard_coefficients,
)

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend

__all__ = ("OpticStudioAnalysis",)


class OpticStudioAnalysis(BaseAnalysis):
    """
    Analyses for the OpticStudio backend.
    """

    def __init__(self, backend: "OpticStudioBackend"):
        super().__init__(backend)
        self._oss = backend.oss

    cardinal_points = _AnalysisMethod(cardinal_points)
    raytrace = _AnalysisMethod(raytrace)
    zernike_standard_coefficients = _AnalysisMethod(zernike_standard_coefficients)
    refraction = _AnalysisMethod(refraction)
