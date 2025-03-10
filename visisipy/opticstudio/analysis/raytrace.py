from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pandas as pd
import zospy as zp

if TYPE_CHECKING:
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
    backend: "type[OpticStudioBackend]",
    coordinates: Iterable[tuple[float, float]],
    wavelengths: Iterable[float] = (0.543,),
    field_type: Literal["angle", "object_height"] = "angle",
    pupil: tuple[float, float] = (0, 0),
) -> tuple[pd.DataFrame, list[zp.analyses.base.AnalysisResult]]:
    """
    Perform a ray trace analysis using the given parameters.
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
    coordinates : Iterable[tuple[float, float]]
        An iterable of tuples representing the coordinates for the ray trace.
        If `field_type` is "angle", the coordinates should be the angles along the (X, Y) axes in degrees.
        If `field_type` is "object_height", the coordinates should be the object heights along the
        (X, Y) axes in mm.
    wavelengths : Iterable[float], optional
        An iterable of wavelengths to be used in the ray trace. Defaults to (0.543,).
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
    backend.set_fields(coordinates, field_type=field_type)
    backend.set_wavelengths(wavelengths)

    raytrace_results = []

    for wavelength_number, wavelength in backend.iter_wavelengths():
        for field_number, field in backend.iter_fields():
            raytrace_result = zp.analyses.raysandspots.single_ray_trace(
                backend.oss,
                px=pupil[0],
                py=pupil[1],
                field=field_number,
                wavelength=wavelength_number,
                global_coordinates=True,
            ).Data.RealRayTraceData

            raytrace_result.insert(0, "Field", [(field.X, field.Y)] * len(raytrace_result))
            raytrace_result.insert(0, "Wavelength", wavelength)

            raytrace_results.append(raytrace_result)

    return _build_raytrace_result(raytrace_results), raytrace_results
