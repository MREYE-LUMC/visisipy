from __future__ import annotations

from typing import TYPE_CHECKING

import zospy as zp

from visisipy.analysis.cardinal_points import CardinalPoints, CardinalPointsResult

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend


def _build_cardinal_points_result(cardinal_points_result: zp.analyses.base.AttrDict) -> CardinalPointsResult:
    return CardinalPointsResult(
        focal_lengths=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Focal Length"],
            object=cardinal_points_result.Data["Object Space"]["Focal Length"],
        ),
        focal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Focal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Focal Planes"],
        ),
        principal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Principal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Principal Planes"],
        ),
        anti_principal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Anti-Principal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Anti-Principal Planes"],
        ),
        anti_nodal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Anti-Nodal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Anti-Nodal Planes"],
        ),
        nodal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Nodal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Nodal Planes"],
        ),
    )


def cardinal_points(
    backend: OpticStudioBackend, surface_1: int | None = None, surface_2: int | None = None
) -> tuple[CardinalPointsResult, zp.analyses.base.AnalysisResult]:
    """
    Get the cardinal points of the system between `surface_1` and `surface_2`.

    Parameters
    ----------
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
    surface_2 = surface_2 or backend.oss.LDE.NumberOfSurfaces - 1

    if surface_1 < 1 or surface_2 > backend.oss.LDE.NumberOfSurfaces - 1:
        raise ValueError("surface_1 and surface_2 must be between 1 and the number of surfaces in the system.")

    if surface_1 >= surface_2:
        raise ValueError("surface_1 must be less than surface_2.")

    cardinal_points_result = zp.analyses.reports.cardinal_points(
        backend.oss,
        surface_1=surface_1,
        surface_2=surface_2,
    )

    return _build_cardinal_points_result(cardinal_points_result), cardinal_points_result
