"""PSF analyses for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import zospy as zp

from visisipy.opticstudio.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    import pandas as pd
    from zospy.analyses.psf.huygens_psf import HuygensPSFResult

    from visisipy.opticstudio import OpticStudioBackend


def fft_psf(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the FFT Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in OpticStudio is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in OpticStudio is used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The field type. Either "angle" or "object_height". Defaults to "angle". This parameter is only used if
        `field_coordinate` is not `None`.
    sampling : SampleSize | str | int, optional
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.
    """

    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type)

    psf_result = zp.analyses.psf.FFTPSF(
        sampling=str(sampling),
        display=str(2 * sampling),
        wavelength=wavelength_number,
        field=field_number,
        psf_type=zp.constants.Analysis.Settings.Psf.FftPsfType.Linear,
        surface="Image",
        normalize=False,
    ).run(backend.get_oss())

    if psf_result.data is None:
        raise ValueError("Failed to run FFT PSF analysis.")

    return psf_result.data, psf_result.data


def huygens_psf(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, HuygensPSFResult]:
    """Calculate the Huygens Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in OpticStudio is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in OpticStudio is used. Defaults to `None`.
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
    HuygensPSFData
        The Huygens PSF result from OpticStudio.
    """

    if not isinstance(pupil_sampling, SampleSize):
        pupil_sampling = SampleSize(pupil_sampling)

    if not isinstance(image_sampling, SampleSize):
        image_sampling = SampleSize(image_sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type)

    psf_result = zp.analyses.psf.HuygensPSFAndStrehlRatio(
        pupil_sampling=str(pupil_sampling),
        image_sampling=str(image_sampling),
        image_delta=0,
        rotation=0,
        wavelength=wavelength_number,
        field=field_number,
        psf_type="Linear",
        show_as="Surface",
        use_polarization=False,
        use_centroid=False,
        normalize=False,
    ).run(backend.get_oss())

    return psf_result.data.psf, psf_result.data


def strehl_ratio(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
) -> tuple[float, HuygensPSFResult]:
    """Calculate the Strehl ratio of the optical system.

    The Strehl ratio is calculated from the point spread function. Which PSF is used depends on the `psf_type` parameter.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the Strehl ratio is calculated. If `None`, the first field coordinate in
        OpticStudio is used.
    wavelength : float | None
        The wavelength at which the Strehl ratio is calculated. If `None`, the first wavelength in OpticStudio is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Defaults to 128.
    psf_type : Literal["fft", "huygens"]
        The type of PSF to be used for the Strehl ratio calculation. Can be either "fft" or "huygens". Defaults to "huygens";
        OpticStudio's FFT PSF does not support calculating the Strehl ratio, so only "huygens" is supported.

    Returns
    -------
    float
        The Strehl ratio of the optical system at the specified field coordinate and wavelength.
    HuygensPSFResult
        The PSF object used to calculate the Strehl ratio. The type of the object depends on the `psf_type` parameter.
    """
    if psf_type == "fft":
        raise NotImplementedError("OpticStudio does not support obtaining the Strehl ratio from the FFT PSF.")

    if psf_type == "huygens":
        _, psf = huygens_psf(
            backend=backend,
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            pupil_sampling=sampling,
            image_sampling=sampling,
        )

        return psf.strehl_ratio, psf

    raise NotImplementedError(f"PSF type '{psf_type}' is not implemented. Only 'huygens' is supported.")
