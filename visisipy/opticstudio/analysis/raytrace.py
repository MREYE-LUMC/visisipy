"""Ray trace analysis for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import zospy as zp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zospy.analyses.raysandspots.single_ray_trace import SingleRayTraceResult

    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.opticstudio.backend import OpticStudioBackend


def _build_raytrace_result(raytrace_results: list[pd.DataFrame]) -> pd.DataFrame:
    columns = {
        "Field": "field",
        "Wavelength": "wavelength",
        "Surf": "surface",
        "Comment": "comment",
        "X-coordinate": "x",
        "Y-coordinate": "y",
        "Z-coordinate": "z",
    }

    return pd.concat(raytrace_results)[columns.keys()].rename(columns=columns).reset_index()


def raytrace(
    backend: type[OpticStudioBackend],
    coordinates: Iterable[FieldCoordinate] | None = None,
    wavelengths: Iterable[float] | None = None,
    field_type: FieldType = "angle",
    pupil: tuple[float, float] = (0, 0),
) -> tuple[pd.DataFrame, list[SingleRayTraceResult]]:
    """Perform a ray trace analysis using the given parameters.
    The ray trace is performed for each wavelength and field in the system, using the Single Ray Trace analysis
    in OpticStudio.

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
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
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
        backend.set_fields(coordinates, field_type=field_type)

    if wavelengths is not None:
        backend.set_wavelengths(wavelengths)

    real_ray_traces = []
    raytrace_results = []

    for wavelength_number, wavelength in backend.iter_wavelengths():
        for field_number, field in backend.iter_fields():
            raytrace_result = zp.analyses.raysandspots.SingleRayTrace(
                px=pupil[0],
                py=pupil[1],
                field=field_number,
                wavelength=wavelength_number,
                global_coordinates=True,
            ).run(backend.get_oss())
            real_ray_trace = raytrace_result.data.real_ray_trace_data

            if real_ray_trace is None:
                raise ValueError(
                    f"Failed to perform ray trace for field ({field.X}, {field.Y}) and wavelength {wavelength}."
                )

            real_ray_trace.insert(0, "Field", [(field.X, field.Y)] * len(real_ray_trace))
            real_ray_trace.insert(0, "Wavelength", wavelength)

            real_ray_traces.append(real_ray_trace)
            raytrace_results.append(raytrace_result.data)

    return _build_raytrace_result(real_ray_traces), raytrace_results
