from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.backend import AnalysisRegistry
from visisipy.opticstudio.analysis.cardinal_points import OpticStudioCardinalPointsMixin
from visisipy.opticstudio.analysis.mtf import OpticStudioMTFMixin
from visisipy.opticstudio.analysis.psf import OpticStudioPSFMixin
from visisipy.opticstudio.analysis.raytrace import OpticStudioRaytraceMixin
from visisipy.opticstudio.analysis.refraction import OpticStudioRefractionMixin
from visisipy.opticstudio.analysis.wavefront import OpticStudioOPDMixin
from visisipy.opticstudio.analysis.zernike_coefficients import OpticStudioZernikeCoefficientsMixin

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend

__all__ = ("OpticStudioAnalysisRegistry",)


class OpticStudioAnalysisRegistry(
    OpticStudioCardinalPointsMixin,
    OpticStudioMTFMixin,
    OpticStudioPSFMixin,
    OpticStudioRaytraceMixin,
    OpticStudioRefractionMixin,
    OpticStudioOPDMixin,
    OpticStudioZernikeCoefficientsMixin,
    AnalysisRegistry["OpticStudioBackend"],
):
    """Analyses for the OpticStudio backend."""

    def __init__(self, backend: OpticStudioBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> OpticStudioBackend:
        return self._backend
