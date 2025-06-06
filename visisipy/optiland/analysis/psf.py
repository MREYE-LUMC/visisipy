from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from optiland.psf import FFTPSF

from visisipy.types import SampleSize

if TYPE_CHECKING:
    from visisipy.optiland import OptilandBackend
    from visisipy.types import FieldCoordinate, FieldType


__all__ = ("fft_psf",)


FFT_PSF_MINIMUM_PUPIL_SAMPLING = 32


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

    return np.floor(32 * np.sqrt(2) ** (np.log2(pupil_sampling) - 5)).astype(int)


def fft_psf(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
) -> tuple[pd.DataFrame, FFTPSF]:

    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    if wavelength is None:
        wavelength = backend.get_wavelengths()[0]
    else:
        backend.set_wavelengths([wavelength])

    normalized_field = backend.get_optic().fields.get_field_coords()[0]
    num_rays = _effective_pupil_sampling(sampling)

    psf = FFTPSF(
        optic=backend.get_optic(),
        field=normalized_field,
        wavelength=wavelength,
        num_rays=num_rays,
        grid_size=int(2 * sampling),
    )

    (psf_extent_x, *_), (psf_extent_y, *_) = psf._get_psf_units(psf.psf)
    index = np.linspace(
        -psf_extent_x / 2, psf_extent_x / 2, psf.psf.shape[0]
    )
    columns = np.linspace(
        -psf_extent_y / 2, psf_extent_y / 2, psf.psf.shape[1]
    )

    df = pd.DataFrame(psf.psf, index=index, columns=columns)

    return df, psf
