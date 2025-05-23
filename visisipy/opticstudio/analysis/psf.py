from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import zospy as zp

from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    import pandas as pd

    from visisipy.opticstudio import OpticStudioBackend


def fft_psf(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    psf_type: Literal["linear", "logarithmic"] = "linear",
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
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 64.
    psf_type : Literal["linear", "logarithmic"], optional
        The PSF normalization type. Either "linear" or "logarithmic". Defaults to "linear".

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.

    Raises
    ------
    ValueError
        If the `psf_type` is not "linear" or "logarithmic".
    """
    if psf_type == "linear":
        psf_type = zp.constants.Analysis.Settings.Psf.FftPsfType.Linear
    elif psf_type == "logarithmic":
        psf_type = zp.constants.Analysis.Settings.Psf.FftPsfType.Log
    else:
        raise ValueError("Invalid PSF type. Must be 'linear' or 'logarithmic'.")

    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    # TODO: create helper for setting wavelengths and fields, because this code is duplicated in other analyses
    wavelength_number = 1 if wavelength is None else backend.get_wavelength_number(wavelength)

    if wavelength_number is None:
        backend.set_wavelengths([wavelength])
        wavelength_number = 1

    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    psf_result = zp.analyses.psf.FFTPSF(
        sampling=str(sampling),
        display=str(2 * sampling),
        wavelength=wavelength_number,
        field=1,
        psf_type=psf_type,
        surface="Image",
        normalize=True,
    ).run(backend.get_oss())

    return psf_result.data, psf_result.data
