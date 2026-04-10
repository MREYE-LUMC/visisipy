"""MTF analyses for OpticStudio."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import zospy as zp

from visisipy.analysis.mtf import MTFResult, SingleMTFResult
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


FIELD_VALUE_REGEX = re.compile(r"-?\d+(?:[.,]\d+)?")


def _parse_field_name(field_name: str) -> FieldCoordinate:
    values = FIELD_VALUE_REGEX.findall(field_name)

    if not values or len(values) > 2:  # noqa: PLR2004
        msg = f"Could not parse field name: {field_name}"
        raise ValueError(msg)

    values = [float(v.replace(",", ".")) for v in values]

    if len(values) == 1:
        return (0.0, values[0])

    return values[0], values[1]


def _build_mtf_result(fft_mtf_result: DataFrame) -> MTFResult:
    if fft_mtf_result.columns.nlevels != 2:  # noqa: PLR2004
        msg = "Expected a MultiIndex with 2 levels for the columns."
        raise ValueError(msg)

    field_names = fft_mtf_result.columns.get_level_values(0).unique()

    result = MTFResult()

    for field_name in field_names:
        field = _parse_field_name(field_name)
        tangential_mtf = fft_mtf_result[field_name, "Tangential"]
        sagittal_mtf = fft_mtf_result[field_name, "Sagittal"]

        result[field] = SingleMTFResult(
            tangential=_transform_mtf_series(tangential_mtf, "tangential"),
            sagittal=_transform_mtf_series(sagittal_mtf, "sagittal"),
        )

    return result


def fft_mtf(
    backend: OpticStudioBackend,
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 128,
    maximum_frequency: float | Literal["default"] = "default",
) -> tuple[MTFResult, DataFrame]:
    """Calculate the FFT Modulation Transfer Function (MTF).

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
    field_coordinate : FieldCoordinate | Literal["all"]
        The field coordinate(s) at which the MTF is calculated. Can be a specific coordinate (e.g., (0, 0)) or
        "all" to calculate for all fields in the backend. Defaults to "all".
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    wavelength : float | None
        The wavelength at which the MTF is calculated. If `None`, the first wavelength in the backend is used.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where
        N is an integer. Only symmetric sample sizes are supported. Defaults to 128.
    maximum_frequency : float | Literal["default"]
        The maximum frequency (in cycles per millimeter) to calculate the MTF up to. If "default", the cutoff frequency is
        determined automatically. Defaults to "default".

    Returns
    -------
    MTFResult
        The MTF data as an MTFResult object, which provides access to the tangential and sagittal MTF values for
        each field coordinate.
    DataFrame
        The raw MTF data as a pandas DataFrame.

    Raises
    ------
    RuntimeError
        If the FFT MTF analysis fails to run or returns no data.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = set_wavelength(backend, wavelength)

    if field_coordinate == "all":
        field_number = "All"
    else:
        field_number = set_field(backend, field_coordinate=field_coordinate, field_type=field_type)

    fft_mtf_result = zp.analyses.mtf.FFTMTF(
        sampling=str(sampling),
        surface="Image",
        wavelength=wavelength_number,
        field=field_number,
        mtf_type=zp.constants.Analysis.Settings.Mtf.MtfTypes.Modulation,
        maximum_frequency=0 if maximum_frequency == "default" else maximum_frequency,
        use_polarization=False,
        use_dashes=False,
    ).run(backend.oss)

    if fft_mtf_result.data is None:
        msg = "Failed to run FFT MTF analysis."
        raise RuntimeError(msg)

    return _build_mtf_result(fft_mtf_result.data), fft_mtf_result.data
