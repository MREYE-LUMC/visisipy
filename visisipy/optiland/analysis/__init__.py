__all__ = ("OptilandAnalysisRegistry",)

from typing import TYPE_CHECKING

from visisipy.backend import BaseAnalysisRegistry

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend


class OptilandAnalysisRegistry(BaseAnalysisRegistry):
    """
    Analyses for the OpticStudio backend.
    """

    def __init__(self, backend: "OptilandBackend"):
        super().__init__(backend)
        self._optic = backend.optic
