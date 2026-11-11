from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.backend import AnalysisRegistry
from visisipy.optiland.analysis.cardinal_points import OptilandCardinalPointsMixin
from visisipy.optiland.analysis.mtf import OptilandMTFMixin
from visisipy.optiland.analysis.psf import OptilandPSFMixin
from visisipy.optiland.analysis.raytrace import OptilandRaytraceMixin
from visisipy.optiland.analysis.refraction import OptilandRefractionMixin
from visisipy.optiland.analysis.wavefront import OptilandOPDMixin
from visisipy.optiland.analysis.zernike_coefficients import OptilandZernikeCoefficientsMixin

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend

__all__ = ("OptilandAnalysisRegistry",)


class OptilandAnalysisRegistry(
    OptilandCardinalPointsMixin,
    OptilandMTFMixin,
    OptilandPSFMixin,
    OptilandRaytraceMixin,
    OptilandRefractionMixin,
    OptilandOPDMixin,
    OptilandZernikeCoefficientsMixin,
    AnalysisRegistry,
):
    """Analyses for the Optiland backend."""

    def __init__(self, backend: OptilandBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> OptilandBackend:
        return self._backend
