from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from visisipy.analysis.base import analysis
from visisipy.backend import get_backend

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pandas import DataFrame

    from visisipy import EyeModel


@analysis
def raytrace(
    model: EyeModel | None,  # noqa: ARG001
    coordinates: Iterable[tuple[float, float]],
    wavelengths: Iterable[float] = (0.543,),
    field_type: Literal["angle", "object"] = "angle",
    pupil: tuple[float, float] = (0, 0),
    *,
    return_raw_result: bool = False,  # noqa: ARG001
) -> DataFrame | tuple[DataFrame, Any]:
    """
    Performs a ray trace analysis using the given parameters.
    The analysis returns a Dataframe with the following columns:

    - field: The field coordinates for the ray trace.
    - wavelength: The wavelength used in the ray trace.
    - surface: The surface number in the system.
    - comment: The comment for the surface.
    - x: The X-coordinate of the ray trace.
    - y: The Y-coordinate of the ray trace.
    - z: The Z-coordinate of the ray trace.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    coordinates : Iterable[tuple[float, float]]
        An iterable of tuples representing the coordinates for the ray trace.
        If `field_type` is "angle", the coordinates should be the angles along the (X, Y) axes in degrees.
        If `field_type` is "object_height", the coordinates should be the object heights along the
        (X, Y) axes in mm.
    wavelengths : Iterable[float], optional
        An iterable of wavelengths to be used in the ray trace. Defaults to (0.543,).
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used in the ray trace. Can be either "angle" or "object_height". Defaults to "angle".
    pupil : tuple[float, float], optional
        A tuple representing the pupil coordinates for the ray trace. Defaults to (0, 0).
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the results of the ray trace analysis.
    Any
        The raw analysis result from the backend.
    """
    return get_backend().analysis.raytrace(
        coordinates=coordinates,
        wavelengths=wavelengths,
        field_type=field_type,
        pupil=pupil,
    )
