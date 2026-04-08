"""Implementation of Visisipy's analyses for the OpticStudio backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.backend import BaseAnalysisRegistry, _AnalysisMethod
from visisipy.opticstudio.analysis.cardinal_points import cardinal_points
from visisipy.opticstudio.analysis.psf import fft_psf, huygens_psf, strehl_ratio
from visisipy.opticstudio.analysis.raytrace import raytrace
from visisipy.opticstudio.analysis.refraction import refraction
from visisipy.opticstudio.analysis.wavefront import opd_map
from visisipy.opticstudio.analysis.zernike_coefficients import (
    zernike_standard_coefficients,
)

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend

__all__ = ("OpticStudioAnalysisRegistry",)


class OpticStudioAnalysisRegistry(BaseAnalysisRegistry):
    """Analyses for the OpticStudio backend."""

    def __init__(self, backend: OpticStudioBackend):
        super().__init__(backend)

    cardinal_points: _AnalysisMethod = _AnalysisMethod(cardinal_points)
    fft_psf: _AnalysisMethod = _AnalysisMethod(fft_psf)
    huygens_psf: _AnalysisMethod = _AnalysisMethod(huygens_psf)
    opd_map: _AnalysisMethod = _AnalysisMethod(opd_map)
    raytrace: _AnalysisMethod = _AnalysisMethod(raytrace)
    refraction: _AnalysisMethod = _AnalysisMethod(refraction)
    strehl_ratio: _AnalysisMethod = _AnalysisMethod(strehl_ratio)
    zernike_standard_coefficients: _AnalysisMethod = _AnalysisMethod(zernike_standard_coefficients)
