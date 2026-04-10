"""MTF analyses for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from optiland.backend import to_numpy
from optiland.mtf import FFTMTF, ScalarFFTMTF, VectorialFFTMTF
from pandas import Series

from visisipy.analysis.mtf import MTFResult, SingleMTFResult
from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.optiland.analysis.psf import _effective_pupil_sampling
from visisipy.types import FieldType, SampleSize

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend
    from visisipy.types import FieldCoordinate


def _build_mtf_result(
    mtf: ScalarFFTMTF | VectorialFFTMTF, fields: FieldCoordinate | list[FieldCoordinate] | Literal["all"]
) -> MTFResult:
    if fields == "all":
        fields = [(f.x, f.y) for f in mtf.optic.fields.fields]
    if isinstance(fields, tuple):
        fields = [fields]

    result = MTFResult()

    for i, field in enumerate(fields):
        tangential_mtf = to_numpy(mtf.mtf[i][0])
        tangential_index = to_numpy(mtf.freq_tang[i])
        sagittal_mtf = to_numpy(mtf.mtf[i][1])
        sagittal_index = to_numpy(mtf.freq_sag[i])

        tangential_series = Series(tangential_mtf, index=tangential_index, name="Tangential MTF")
        tangential_series.index.name = "frequency"
        sagittal_series = Series(sagittal_mtf, index=sagittal_index, name="Sagittal MTF")
        sagittal_series.index.name = "frequency"

        result[field] = SingleMTFResult(
            tangential=tangential_series,
            sagittal=sagittal_series,
        )

    return result


def fft_mtf(
    backend: OptilandBackend,
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 128,
    maximum_frequency: float | Literal["default"] = "default",
) -> tuple[MTFResult, ScalarFFTMTF | VectorialFFTMTF]:
    """Calculate the FFT Modulation Transfer Function (MTF).

    Parameters
    ----------
    backend : OptilandBackend
        Reference to the Optiland backend.
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
    ScalarFFTMTF | VectorialFFTMTF
        The raw MTF result object from the optiland backend. If the optic has a polarization state, a VectorialFFTMTF is
        returned, otherwise a ScalarFFTMTF is returned.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength = set_wavelength(backend, wavelength)

    if field_coordinate == "all":
        normalized_field = "all"
    else:
        normalized_field = [set_field(backend, field_coordinate=field_coordinate, field_type=field_type)]

    mtf: ScalarFFTMTF | VectorialFFTMTF = FFTMTF(
        optic=backend.optic,
        fields=normalized_field,
        wavelength=wavelength,
        num_rays=_effective_pupil_sampling(sampling),
        grid_size=int(2 * sampling),
        max_freq="cutoff" if maximum_frequency == "default" else maximum_frequency,
    )

    return _build_mtf_result(mtf, field_coordinate), mtf
