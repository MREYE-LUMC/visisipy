from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from optiland.wavefront import OPD
from pandas import DataFrame

from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend


def opd_map(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
    use_exit_pupil_shape: bool = False,
) -> tuple[DataFrame, OPD]:
    """Calculate the Optical Path Difference (OPD) map at the retina surface.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to use for the wavefront calculation. If `None`, the currently built model will be used.
    field_coordinate : FieldCoordinate | None
        The coordinate of the field for which the wavefront is calculated. If `None`, the current field coordinate will be used.
    wavelength : float | None
        The wavelength (in nm) for which the wavefront is calculated. If `None`, the current wavelength will be used.
    field_type : FieldType
        The type of field coordinate provided. Either 'angle' (degrees) or 'object_height' (mm). Defaults to 'angle'.
    sampling : SampleSize | str | int
        The sampling of the OPD map. Can be an integer (e.g., 128 for 128x128), a string like '128x128', or a `SampleSize` object.
        Defaults to 128.
    remove_tilt : bool, optional
        If `True`, the tilt component is removed from the OPD map. Defaults to `True`.
    use_exit_pupil_shape : bool, optional
        If `True`, the OPD map is distorted to show the shape of the exit pupil. Defaults to `False`. Optiland does not support this
        feature. A warning will be issued if set to `True`, and the parameter will be ignored.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the OPD map values in waves.
    OPD
        The raw analysis result from the backend.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    if use_exit_pupil_shape:
        warn("Correcting for the exit pupil shape is not supported in Optiland.", UserWarning, stacklevel=2)

    wavelength = set_wavelength(backend, wavelength)
    normalized_field = set_field(backend, field_coordinate, field_type)

    opd_result = OPD(
        optic=backend.get_optic(),
        wavelength=wavelength,
        field=normalized_field,
        num_rays=int(sampling),
        distribution="uniform",
        strategy="chief_ray",
        remove_tilt=remove_tilt,
    )

    data = opd_result.generate_opd_map(num_points=int(sampling))

    opd_dataframe = DataFrame(
        data["z"],
        index=data["y"][:, 0],
        columns=data["x"][0, :],
        dtype=float,
    )

    return opd_dataframe, opd_result
