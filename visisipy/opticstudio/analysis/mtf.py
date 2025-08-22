from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import zospy as zp

from visisipy.analysis.mtf import MTFResult
from visisipy.opticstudio.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldType, SampleSize

if TYPE_CHECKING:
    from pandas import DataFrame, Series

    from visisipy.opticstudio.backend import OpticStudioBackend
    from visisipy.types import FieldCoordinate


def _transform_mtf_series(series: Series, direction: Literal["sagittal", "tangential"]) -> Series:
    series.index.name = "frequency"
    series.name = f"{direction.title()} MTF"

    return series


def fft_mtf(
    backend: type[OpticStudioBackend],
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | None = None,
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
) -> tuple[MTFResult, DataFrame]:
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type=field_type)

    fft_mtf_result = zp.analyses.mtf.FFTMTF(
        sampling=str(sampling),
        surface="Image",
        wavelength=wavelength_number,
        field=field_number,
        mtf_type=zp.constants.Analysis.Settings.Mtf.MtfTypes.Modulation,
        maximum_frequency=0 if maximum_frequency == "default" else maximum_frequency,
        use_polarization=False,
        use_dashes=False,
    ).run(backend.get_oss())

    if fft_mtf_result.data is None:
        raise RuntimeError("Failed to run FFT MTF analysis.")

    field_name = fft_mtf_result.data.columns[0][0]

    tangential_mtf = fft_mtf_result.data[(field_name, "Tangential")]
    sagittal_mtf = fft_mtf_result.data[(field_name, "Sagittal")]

    return MTFResult(
        tangential=_transform_mtf_series(tangential_mtf, "tangential"),
        sagittal=_transform_mtf_series(sagittal_mtf, "sagittal"),
    ), fft_mtf_result.data
