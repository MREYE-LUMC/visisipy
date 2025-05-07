"""Cardinal points analysis for Optiland."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from visisipy.analysis.cardinal_points import CardinalPoints, CardinalPointsResult

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial

    from visisipy.optiland import OptilandBackend


__all__ = ("cardinal_points",)


def _build_cardinal_points_result(paraxial: Paraxial) -> CardinalPointsResult:
    return CardinalPointsResult(
        focal_lengths=CardinalPoints(
            object=paraxial.f1(),
            image=paraxial.f2(),
        ),
        focal_points=CardinalPoints(
            object=paraxial.F1(),
            image=paraxial.F2(),
        ),
        principal_points=CardinalPoints(
            object=paraxial.P1(),
            image=paraxial.P2(),
        ),
        anti_principal_points=CardinalPoints(
            object=paraxial.P1anti(),
            image=paraxial.P2anti(),
        ),
        nodal_points=CardinalPoints(
            object=paraxial.N1(),
            image=paraxial.N2(),
        ),
        anti_nodal_points=CardinalPoints(
            object=paraxial.N1anti(),
            image=paraxial.N2anti(),
        ),
    )


def cardinal_points(
    backend: OptilandBackend, surface_1: int | None = None, surface_2: int | None = None
) -> tuple[CardinalPointsResult, Paraxial]:
    """Get the cardinal points of the system between `surface_1` and `surface_2`.

    Note that Optiland only supports calculating cardinal points for the entire system, not for a subset of surfaces.
    A ValueError will be raised if `surface_1` or `surface_2` are different from the first and last surfaces in the system.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the Optiland backend.
    surface_1 : int | None, optional
        The first surface to be used in the analysis. If `None`, the first surface in the system will be used.
        Defaults to `None`.
    surface_2 : int | None, optional
        The second surface to be used in the analysis. If `None`, the last surface in the system will be used.
        Defaults to `None`.

    Returns
    -------
    CardinalPointsResult
        The cardinal points of the system.

    Raises
    ------
    ValueError
        If `surface_1` or `surface_2` are not between 1 and the number of surfaces in the system, or if `surface_1`
        is greater than or equal to `surface_2`.
    """
    if (surface_1 is not None and surface_1 not in {0, 1}) or (
        surface_2 is not None and surface_2 != backend.get_optic().surface_group.num_surfaces - 1
    ):
        raise ValueError("Optiland only supports calculating cardinal points for the entire system.")

    return _build_cardinal_points_result(backend.get_optic().paraxial), deepcopy(backend.get_optic().paraxial)
