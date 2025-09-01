"""PSF analyses for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from optiland.psf import FFTPSF, HuygensPSF

from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from visisipy.optiland import OptilandBackend
    from visisipy.types import FieldCoordinate, FieldType


__all__ = ("fft_psf", "huygens_psf")


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
    sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, FFTPSF]:
    """Calculate the FFT Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the Optiland backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in Optiland is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in Optiland is used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The field type. Either "angle" or "object_height". Defaults to "angle". This parameter is only used if
        `field_coordinate` is not `None`.
    sampling : SampleSize | str | int, optional
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.
    FFTPSF
        The Optiland FFTPSF object.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    normalized_field = set_field(backend, field_coordinate, field_type)
    wavelength = set_wavelength(backend, wavelength)

    num_rays = _effective_pupil_sampling(sampling)

    psf = FFTPSF(
        optic=backend.get_optic(),
        field=normalized_field,
        wavelength=wavelength,
        num_rays=num_rays,
        grid_size=int(2 * sampling),
    )

    (psf_extent_x, *_), (psf_extent_y, *_) = psf._get_psf_units(psf.psf)  # noqa: SLF001
    index = np.linspace(-psf_extent_x / 2, psf_extent_x / 2, psf.psf.shape[0])
    columns = np.linspace(-psf_extent_y / 2, psf_extent_y / 2, psf.psf.shape[1])

    # The PSF rows are reversed in the y-direction to match the orientation of the PSF in OpticStudio.
    df = pd.DataFrame(psf.psf / 100, index=index, columns=columns)

    return df, psf


def huygens_psf(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, HuygensPSF]:
    """Calculate the Huygens Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the Optiland backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in Optiland is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in Optiland is used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The field type. Either "angle" or "object_height". Defaults to "angle". This parameter is only used if
        `field_coordinate` is not `None`.
    pupil_sampling : SampleSize | str | int, optional
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.
    image_sampling : SampleSize | str | int, optional
        The size of the PSF grid, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.
    HuygensPSF
        The Optiland HuygensPSF object.
    """
    if not isinstance(pupil_sampling, SampleSize):
        pupil_sampling = SampleSize(pupil_sampling)

    if not isinstance(image_sampling, SampleSize):
        image_sampling = SampleSize(image_sampling)

    normalized_field = set_field(backend, field_coordinate, field_type)
    wavelength = set_wavelength(backend, wavelength)

    psf = HuygensPSF(
        optic=backend.get_optic(),
        field=normalized_field,
        wavelength=wavelength,
        num_rays=int(pupil_sampling),
        image_size=int(image_sampling),
    )

    (psf_extent_x, *_), (psf_extent_y, *_) = psf._get_psf_units(psf.psf)  # noqa: SLF001
    index = np.linspace(-psf_extent_x / 2, psf_extent_x / 2, psf.psf.shape[0])
    columns = np.linspace(-psf_extent_y / 2, psf_extent_y / 2, psf.psf.shape[1])

    # The PSF rows are reversed in the y-direction to match the orientation of the PSF in OpticStudio.
    df = pd.DataFrame(psf.psf / 100, index=index, columns=columns)

    return df, psf


def strehl_ratio(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
) -> tuple[float, FFTPSF | HuygensPSF]:
    """Calculate the Strehl ratio of the optical system.

    The Strehl ratio is calculated from the point spread function. Which PSF is used depends on the `psf_type` parameter.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the Optiland backend.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the Strehl ratio is calculated. If `None`, the first field coordinate in
        Optiland is used.
    wavelength : float | None
        The wavelength at which the Strehl ratio is calculated. If `None`, the first wavelength in Optiland is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Defaults to 128.
    psf_type : Literal["fft", "huygens"]
        The type of PSF to be used for the Strehl ratio calculation. Can be either "fft" or "huygens". Defaults to "huygens".

    Returns
    -------
    float
        The Strehl ratio of the optical system at the specified field coordinate and wavelength.
    FFTPSF | HuygensPSF
        The PSF object used to calculate the Strehl ratio. The type of the object depends on the `psf_type` parameter.
    """
    if psf_type == "fft":
        _, psf = fft_psf(
            backend=backend,
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
        )

        return psf.strehl_ratio(), psf

    if psf_type == "huygens":
        _, psf = huygens_psf(
            backend=backend,
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            pupil_sampling=sampling,
            image_sampling=sampling,
        )

        return psf.strehl_ratio(), psf

    raise NotImplementedError(f"PSF type '{psf_type}' is not implemented.")
