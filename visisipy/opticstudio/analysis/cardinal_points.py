"""Cardinal points analysis for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zospy as zp

from visisipy.analysis.cardinal_points import CardinalPoints, CardinalPointsResult

if TYPE_CHECKING:
    from zospy.analyses.reports.cardinal_points import CardinalPointsResult as ZOSPyCardinalPointsResult

    from visisipy.opticstudio.backend import OpticStudioBackend


def _build_cardinal_points_result(cardinal_points_result: ZOSPyCardinalPointsResult) -> CardinalPointsResult:
    return CardinalPointsResult(
        focal_lengths=CardinalPoints(
            image=cardinal_points_result.cardinal_points.focal_length.image,
            object=cardinal_points_result.cardinal_points.focal_length.object,
        ),
        focal_points=CardinalPoints(
            image=cardinal_points_result.cardinal_points.focal_planes.image,
            object=cardinal_points_result.cardinal_points.focal_planes.object,
        ),
        principal_points=CardinalPoints(
            image=cardinal_points_result.cardinal_points.principal_planes.image,
            object=cardinal_points_result.cardinal_points.principal_planes.object,
        ),
        anti_principal_points=CardinalPoints(
            image=cardinal_points_result.cardinal_points.anti_principal_planes.image,
            object=cardinal_points_result.cardinal_points.anti_principal_planes.object,
        ),
        nodal_points=CardinalPoints(
            image=cardinal_points_result.cardinal_points.nodal_planes.image,
            object=cardinal_points_result.cardinal_points.nodal_planes.object,
        ),
        anti_nodal_points=CardinalPoints(
            image=cardinal_points_result.cardinal_points.anti_nodal_planes.image,
            object=cardinal_points_result.cardinal_points.anti_nodal_planes.object,
        ),
    )


def cardinal_points(
    backend: OpticStudioBackend, surface_1: int | None = None, surface_2: int | None = None
) -> tuple[CardinalPointsResult, ZOSPyCardinalPointsResult]:
    """Get the cardinal points of the system between `surface_1` and `surface_2`.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
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
    surface_1 = surface_1 or 1
    surface_2 = surface_2 or backend.get_oss().LDE.NumberOfSurfaces - 1

    if surface_1 < 1 or surface_2 > backend.get_oss().LDE.NumberOfSurfaces - 1:
        raise ValueError("surface_1 and surface_2 must be between 1 and the number of surfaces in the system.")

    if surface_1 >= surface_2:
        raise ValueError("surface_1 must be less than surface_2.")

    cardinal_points_result = zp.analyses.reports.CardinalPoints(
        surface_1=surface_1,
        surface_2=surface_2,
    ).run(backend.get_oss())

    return _build_cardinal_points_result(cardinal_points_result.data), cardinal_points_result.data
