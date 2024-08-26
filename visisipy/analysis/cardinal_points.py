from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from visisipy.analysis.base import analysis
from visisipy.backend import get_backend

if TYPE_CHECKING:
    from visisipy import EyeModel


__all__ = ("cardinal_points", "CardinalPoints", "CardinalPointsResult")


class CardinalPoints(NamedTuple):
    """
    The cardinal points of a system in object and image space.
    """

    object: float
    image: float


@dataclass
class CardinalPointsResult:
    """
    The cardinal points of a system.

    Attributes
    ----------
    focal_lengths : CardinalPoints
        The focal lengths of the system.
    focal_points : CardinalPoints
        The focal points of the system.
    principal_points : CardinalPoints
        The principal points of the system.
    anti_principal_points : CardinalPoints
        The anti-principal points of the system.
    nodal_points : CardinalPoints
        The nodal points of the system.
    anti_nodal_points : CardinalPoints
        The anti-nodal points of the system.
    """

    focal_lengths: CardinalPoints
    focal_points: CardinalPoints
    principal_points: CardinalPoints
    anti_principal_points: CardinalPoints
    nodal_points: CardinalPoints
    anti_nodal_points: CardinalPoints


@analysis
def cardinal_points(
    model: EyeModel | None,  # noqa: ARG001
    surface_1: int | None = None,
    surface_2: int | None = None,
    *,
    return_raw_result: bool = False,  # noqa: ARG001
) -> CardinalPointsResult | tuple[CardinalPointsResult, Any]:
    """
    Get the cardinal points of the system between `surface_1` and `surface_2`.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    surface_1 : int | None, optional
        The first surface to be used in the analysis. If `None`, the first surface in the system will be used.
        Defaults to `None`.
    surface_2 : int | None, optional
        The second surface to be used in the analysis. If `None`, the last surface in the system will be used.
        Defaults to `None`.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.

    Returns
    -------
    CardinalPointsResult
        The cardinal points of the system.
    Any
        The raw analysis result from the backend.
    """
    return get_backend().analysis.cardinal_points(
        surface_1=surface_1,
        surface_2=surface_2,
    )
