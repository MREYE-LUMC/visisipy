"""Ray trace analysis for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

    from optiland.optic import Optic

    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.optiland import OptilandBackend

__all__ = ("raytrace",)


def _trace_single_ray(
    optic: Optic,
    field: FieldCoordinate,
    pupil: tuple[float, float],
    wavelength: float,
) -> pd.DataFrame:
    optic.trace_generic(*field, *pupil, wavelength=wavelength)
    x = optic.surface_group.x
    y = optic.surface_group.y
    z = optic.surface_group.z
    surface_numbers = range(optic.surface_group.num_surfaces)

    return pd.DataFrame(
        {
            "wavelength": [wavelength] * len(x),
            "surface": surface_numbers,
            "comment": [None] * len(x),
            "x": x[:, 0],
            "y": y[:, 0],
            "z": z[:, 0],
        }
    )


def raytrace(
    backend: type[OptilandBackend],
    coordinates: Iterable[FieldCoordinate] | None = None,
    wavelengths: Iterable[float] | None = None,
    field_type: FieldType = "angle",
    pupil: tuple[float, float] = (0, 0),
) -> tuple[pd.DataFrame, None]:
    """Perform a ray trace analysis using the given parameters.

    The ray trace is performed for each wavelength and field in the system, using the generic ray trace in Optiland.

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
    backend : type[OptilandBackend]
        The Optiland backend to use for the ray trace.
    coordinates : Iterable[tuple[float, float]], optional
        An iterable of tuples representing the coordinates for the ray trace.
        If `field_type` is "angle", the coordinates should be the angles along the (X, Y) axes in degrees.
        If `field_type` is "object_height", the coordinates should be the object heights along the
        (X, Y) axes in mm. Defaults to `None`, which uses the fields defined in the backend.
    wavelengths : Iterable[float], optional
        An iterable of wavelengths to be used in the ray trace. Defaults to `None`, which uses the wavelengths
        defined in the backend.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used in the ray trace. Can be either "angle" or "object_height". Defaults to
        "angle".
    pupil : tuple[float, float], optional
        A tuple representing the pupil coordinates for the ray trace. Defaults to (0, 0).

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the results of the ray trace analysis.
    """
    if abs(pupil[0]) > 1 or abs(pupil[1]) > 1:
        raise ValueError("Pupil coordinates must be between -1 and 1.")

    if coordinates is not None:
        backend.set_fields(coordinates, field_type)

    if wavelengths is not None:
        backend.set_wavelengths(wavelengths)

    raytrace_results = []

    normalized_fields = backend.get_optic().fields.get_field_coords()

    for _, wavelength in backend.iter_wavelengths():
        for (_, field), normalized_field in zip(backend.iter_fields(), normalized_fields):
            result = _trace_single_ray(backend.get_optic(), normalized_field, pupil, wavelength)
            result.insert(0, "field", [field] * len(result))
            raytrace_results.append(result)

    return pd.concat(raytrace_results).reset_index(), None
