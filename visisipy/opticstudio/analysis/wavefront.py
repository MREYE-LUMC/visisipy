"""Wavefront analysis for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zospy as zp
from pandas import DataFrame

from visisipy.opticstudio.analysis.helpers import set_field, set_wavelength
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from visisipy.opticstudio.backend import OpticStudioBackend
    from visisipy.types import FieldCoordinate, FieldType


def _pad_index(index: NDArray) -> NDArray:
    """Pad the index of the DataFrame to include the outer edge.

    Parameters
    ----------
    index : NDArray
        The index to pad.

    Returns
    -------
    NDArray
        The padded index.
    """

    step = index[1] - index[0]
    new_index = index[0] - step

    return np.insert(index, 0, new_index)


def _pad_opd_map(data: DataFrame) -> DataFrame:
    """Pad the OPD map with an empty row and column at the left and bottom edges.

    To ensure that the center of the wavefront is sampled, OpticStudio samples one row and
    column outside the pupil. This effectively yields a grid that is (N-1) x (N-1) for an NxN sampling.
    ZOSPy returns the data without this outer row and column, but to be consistent with other backends,
    and to ensure that the output shape equals the requested sampling, this row and column are added back.

    Parameters
    ----------
    data : DataFrame
        The OPD map data as a pandas DataFrame.

    Returns
    -------
    DataFrame
        The padded OPD map as a pandas DataFrame.
    """
    x_coordinates = data.columns.to_numpy()
    y_coordinates = data.index.to_numpy()

    padded_opd = np.pad(data.to_numpy(), pad_width=((1, 0), (1, 0)), mode="constant", constant_values=np.nan)

    return DataFrame(
        padded_opd,
        index=_pad_index(y_coordinates),
        columns=_pad_index(x_coordinates),
        dtype=float,
    )


def opd_map(
    backend: OpticStudioBackend,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
    use_exit_pupil_shape: bool = False,
) -> tuple[DataFrame, DataFrame]:
    """Calculate the Optical Path Difference (OPD) map at the retina surface.

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
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
        If `True`, the OPD map is distorted to show the shape of the exit pupil. Defaults to `False`.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the OPD map values in waves.
    DataFrame
        The raw analysis result from the backend.

    Raises
    ------
    RuntimeError
        If the wavefront analysis fails or returns no data.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type)

    wavefront_result = zp.analyses.wavefront.WavefrontMap(
        field=field_number,
        surface="Image",
        wavelength=wavelength_number,
        show_as=zp.constants.Analysis.ShowAs.Surface,
        rotation=zp.constants.Analysis.Settings.Rotations.Rotate_0,
        sampling=str(sampling),
        polarization=zp.constants.Analysis.Settings.Polarizations.None_,
        reference_to_primary=False,
        use_exit_pupil=use_exit_pupil_shape,
        remove_tilt=remove_tilt,
        scale=1.0,
        sub_aperture_x=0.0,
        sub_aperture_y=0.0,
        sub_aperture_r=1.0,
        contour_format="",
    ).run(backend.oss)

    if wavefront_result.data is None:
        raise RuntimeError("Wavefront analysis failed.")

    return _pad_opd_map(wavefront_result.data), wavefront_result.data
