from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from optiland.wavefront import OPD
from optiland.wavefront.opd import OPDData
from pandas import DataFrame

from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    from optiland.distribution import BaseDistribution
    from optiland.wavefront import WavefrontData

    from visisipy.optiland.backend import OptilandBackend


__all__ = ("opd_map",)


def generate_opd_map(wavefront: WavefrontData, distribution: BaseDistribution, sampling: int) -> OPDData:
    """Fast generation of an OPD map from wavefront data with uniform sampling.

    Optiland's `OPD.generate_opd_map` method uses cubic grid interpolation, which is unnecessarily slow
    for wavefront data generated with a uniform distribution. This function provides a faster alternative
    by using direct indexing.

    Parameters
    ----------
    wavefront : WavefrontData
        The wavefront data containing OPD values.
    distribution : BaseDistribution
        The pupil sampling distribution used for the wavefront.
    sampling : int
        The desired sampling size of the OPD map (e.g., 128 for a 128x128 map).

    Returns
    -------
    OPDData
        A dictionary containing the x and y coordinates and the OPD map (z values).
    """
    pupil_x = np.linspace(-1, 1, sampling)
    pupil_y = np.linspace(-1, 1, sampling)

    opd_map = np.full((sampling, sampling), np.nan)

    x_indices = np.searchsorted(pupil_x, distribution.x)
    y_indices = np.searchsorted(pupil_y, distribution.y)

    # Due to the use of meshgrid in Optilands opd map generation, the x-direction is along the columns
    opd_map[y_indices, x_indices] = wavefront.opd

    x_opd, y_opd = np.meshgrid(
        pupil_x,
        pupil_y,
    )

    return OPDData(
        x=x_opd,
        y=y_opd,
        z=opd_map,
    )


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

    data = generate_opd_map(
        opd_result.get_data(field=normalized_field, wl=wavelength),
        distribution=opd_result.distribution,
        sampling=int(sampling),
    )

    opd_dataframe = DataFrame(
        data["z"],
        index=data["y"][:, 0],
        columns=data["x"][0, :],
        dtype=float,
    )

    return opd_dataframe, opd_result
