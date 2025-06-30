"""Calculate the point spread function (PSF) of the eye."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy import EyeModel
    from visisipy.backend import BaseBackend
    from visisipy.types import FieldCoordinate, FieldType, SampleSize


@overload
def fft_psf(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame: ...


@overload
def fft_psf(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[DataFrame, Any]: ...


@analysis
def fft_psf(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame | tuple[DataFrame, Any]:
    """Calculate the FFT Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the PSF is calculated. If `None`, the first field coordinate in the backend is
        used.
    wavelength : float | None
        The wavelength at which the PSF is calculated. If `None`, the first wavelength in the backend is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Only symmetric sample sizes are supported. Defaults to 128.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
    DataFrame
        The PSF data as a DataFrame. The DataFrame contains the PSF values at the specified field coordinate and
        wavelength.
    """
    return backend.analysis.fft_psf(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=sampling,
    )


@overload
def huygens_psf(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame: ...


@overload
def huygens_psf(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[DataFrame, Any]: ...


@analysis
def huygens_psf(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame | tuple[DataFrame, Any]:
    """Calculate the Huygens Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the analysis. If `None`, the current eye model will be used.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the PSF is calculated. If `None`, the first field coordinate in the backend is
        used.
    wavelength : float | None
        The wavelength at which the PSF is calculated. If `None`, the first wavelength in the backend is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    pupil_sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Only symmetric sample sizes are supported. Defaults to 128.
    image_sampling : SampleSize | str | int
        The size of the PSF grid. Can be an integer or a string in the format "NxN", where N is an integer. Only
        symmetric sample sizes are supported. Defaults to 128.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
    DataFrame
        The PSF data as a DataFrame. The DataFrame contains the PSF values at the specified field coordinate and
        wavelength.
    """
    return backend.analysis.huygens_psf(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        pupil_sampling=pupil_sampling,
        image_sampling=image_sampling,
    )


@overload
def strehl_ratio(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> float: ...


@overload
def strehl_ratio(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[float, Any]: ...


@analysis
def strehl_ratio(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> float | tuple[float, Any]:
    """Calculate the Strehl ratio of the optical system.

    The Strehl ratio is calculated from the point spread function. Which PSF is used depends on the `psf_type` parameter.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the Strehl ratio is calculated. If `None`, the first field coordinate in the
        backend is used.
    wavelength : float | None
        The wavelength at which the Strehl ratio is calculated. If `None`, the first wavelength in the backend is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Only symmetric sample sizes are supported. Defaults to 128.
    psf_type : Literal["fft", "huygens"]
        The type of PSF to be used for the Strehl ratio calculation. Can be either "fft" or "huygens". Defaults to "huygens".
        Not all psf types are supported by all backends.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
    float
        The Strehl ratio of the optical system at the specified field coordinate and wavelength.
    """
    if psf_type not in {"fft", "huygens"}:
        raise ValueError(f"Invalid psf_type: {psf_type}. Must be 'fft' or 'huygens'.")

    return backend.analysis.strehl_ratio(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=sampling,
        psf_type=psf_type,
    )
