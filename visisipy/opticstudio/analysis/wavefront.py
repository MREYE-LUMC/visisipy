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
    """Pad the index of the DataFrame to include the outer edge."""

    step = index[1] - index[0]
    new_index = index[0] - step

    return np.insert(index, 0, new_index)


def _pad_opd_map(data: DataFrame) -> DataFrame:
    """Pad the OPD map with an empty row and column at the left and bottom edges.

    To ensure that the center of the wavefront is sampled, OpticStudio samples one row and
    column outside the pupil. This effectively yields a grid that is (N-1) x (N-1) for an NxN sampling.
    ZOSPy returns the data without this outer row and column, but to be consistent with other backends,
    and to ensure that the output shape equals the requested sampling, this row and column are added back.
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
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
) -> tuple[DataFrame, DataFrame]:
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
        use_exit_pupil=False,
        remove_tilt=remove_tilt,
        scale=1.0,
        sub_aperture_x=0.0,
        sub_aperture_y=0.0,
        sub_aperture_r=1.0,
        contour_format="",
    ).run(backend.get_oss())

    if wavefront_result.data is None:
        raise RuntimeError("Wavefront analysis failed.")

    return _pad_opd_map(wavefront_result.data), wavefront_result.data
