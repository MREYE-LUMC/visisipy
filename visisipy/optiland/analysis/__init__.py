"""Analyze optical eye models in Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.backend import BaseAnalysisRegistry, _AnalysisMethod
from visisipy.optiland.analysis.cardinal_points import cardinal_points
from visisipy.optiland.analysis.mtf import fft_mtf
from visisipy.optiland.analysis.psf import fft_psf, huygens_psf, strehl_ratio
from visisipy.optiland.analysis.raytrace import raytrace
from visisipy.optiland.analysis.refraction import refraction
from visisipy.optiland.analysis.wavefront import opd_map
from visisipy.optiland.analysis.zernike_coefficients import zernike_standard_coefficients

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend

__all__ = ("OptilandAnalysisRegistry",)


class OptilandAnalysisRegistry(BaseAnalysisRegistry):
    """Analyses for the OpticStudio backend."""

    def __init__(self, backend: OptilandBackend):
        super().__init__(backend)

    cardinal_points: _AnalysisMethod = _AnalysisMethod(cardinal_points)  # pyrefly: ignore[bad-override]
    fft_mtf: _AnalysisMethod = _AnalysisMethod(fft_mtf)  # pyrefly: ignore[bad-override]
    fft_psf: _AnalysisMethod = _AnalysisMethod(fft_psf)  # pyrefly: ignore[bad-override]
    huygens_psf: _AnalysisMethod = _AnalysisMethod(huygens_psf)  # pyrefly: ignore[bad-override]
    opd_map: _AnalysisMethod = _AnalysisMethod(opd_map)  # pyrefly: ignore[bad-override]
    raytrace: _AnalysisMethod = _AnalysisMethod(raytrace)  # pyrefly: ignore[bad-override]
    refraction: _AnalysisMethod = _AnalysisMethod(refraction)  # pyrefly: ignore[bad-override]
    strehl_ratio: _AnalysisMethod = _AnalysisMethod(strehl_ratio)  # pyrefly: ignore[bad-override]
    zernike_standard_coefficients: _AnalysisMethod = _AnalysisMethod(zernike_standard_coefficients)  # pyrefly: ignore[bad-override]
