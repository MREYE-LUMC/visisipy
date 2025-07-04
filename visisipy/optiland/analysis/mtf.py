from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from optiland.mtf import FFTMTF
from pandas import Series

from visisipy.analysis.mtf import MTFResult
from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldType, SampleSize

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend
    from visisipy.types import FieldCoordinate


FFT_PSF_MINIMUM_PUPIL_SAMPLING = 32


def _build_mtf_result(mtf: FFTMTF) -> MTFResult:
    tangential_mtf = mtf.mtf[0][0]
    sagittal_mtf = mtf.mtf[0][1]

    index = np.arange(len(tangential_mtf)) * mtf._get_mtf_units()  # noqa: SLF001

    tangential_series = Series(tangential_mtf, index=index, name="Tangential MTF")
    tangential_series.index.name = "frequency"
    sagittal_series = Series(sagittal_mtf, index=index, name="Sagittal MTF")
    sagittal_series.index.name = "frequency"

    return MTFResult(
        tangential=tangential_series,
        sagittal=sagittal_series,
    )


def _effective_pupil_sampling(pupil_sampling: int | SampleSize) -> int:
    """Calculate the effective pupil sampling based on the given pupil sampling.
    The rationale behind this calculation is documented in the OpticStudio documentation [1]_.
    Calculating the effective pupil sampling ensures that the PSF extent is similar to OpticStudio's PSF extent.
    See also this discussion on GitHub: https://github.com/HarrisonKramer/optiland/discussions/157
    .. [1] https://ansyshelp.ansys.com/public/account/secured?returnurl=/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/FFT_PSF.html
    Parameters
    ----------
    pupil_sampling : int
        The pupil sampling value to calculate the effective pupil sampling from.
    Returns
    -------
    int
        The effective pupil sampling value, which is the number of rays used to sample the wavefront in the exit pupil.
    """
    pupil_sampling = int(pupil_sampling)

    if pupil_sampling < FFT_PSF_MINIMUM_PUPIL_SAMPLING:
        raise ValueError("Pupil sampling must be at least 32 for FFT PSF calculation.")

    return np.floor(32 * 2 ** ((np.log2(pupil_sampling) - 5) / 2)).astype(int)


def fft_mtf(
    backend: type[OptilandBackend],
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | None = None,
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
) -> tuple[MTFResult, FFTMTF]:
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength = set_wavelength(backend, wavelength)
    normalized_field = set_field(backend, field_coordinate=field_coordinate, field_type=field_type)

    mtf = FFTMTF(
        optic=backend.get_optic(),
        fields=[normalized_field],
        wavelength=wavelength,
        num_rays=_effective_pupil_sampling(sampling),
        grid_size=2 * int(sampling),
        max_freq="cutoff" if maximum_frequency == "default" else maximum_frequency,
    )

    return _build_mtf_result(mtf), mtf
