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
        raise ValueError(f"Could not parse field name: {field_name}")

    values = [float(v.replace(",", ".")) for v in values]

    if len(values) == 1:
        return (0.0, values[0])

    return values[0], values[1]


def _build_mtf_result(fft_mtf_result: DataFrame) -> MTFResult:
    if fft_mtf_result.columns.nlevels != 2:  # noqa: PLR2004
        raise ValueError("Expected a MultiIndex with 2 levels for the columns.")

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
    backend: type[OpticStudioBackend],
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
) -> tuple[MTFResult, DataFrame]:
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
    ).run(backend.get_oss())

    if fft_mtf_result.data is None:
        raise RuntimeError("Failed to run FFT MTF analysis.")

    return _build_mtf_result(fft_mtf_result.data), fft_mtf_result.data
