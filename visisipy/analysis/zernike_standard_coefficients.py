"""Calculate the Zernike standard coefficients for an eye model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis
from visisipy.types import FieldType, SampleSize
from visisipy.wavefront import min_max_noll_index

if TYPE_CHECKING:
    from visisipy.backend import BaseBackend
    from visisipy.models import EyeModel
    from visisipy.wavefront import ZernikeCoefficients

__all__ = (
    "rms_hoa",
    "zernike_standard_coefficients",
)


@overload
def zernike_standard_coefficients(
    model: EyeModel | None = None,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> ZernikeCoefficients: ...


@overload
def zernike_standard_coefficients(
    model: EyeModel | None = None,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[ZernikeCoefficients, Any]: ...


@analysis
def zernike_standard_coefficients(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[ZernikeCoefficients, Any]:
    """Calculates the Zernike standard coefficients at the retina surface.

    Zernike standard coefficients are returned in the Noll notation.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate for the Zernike calculation. When `None`, the first field in the backend is used.
        Defaults to `None`.
    wavelength : float | None, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in the backend is used.
        Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    sampling : SampleSize | str | int, optional
        The sampling for the Zernike calculation. Defaults to 64.
    maximum_term : int, optional
        The maximum term for the Zernike calculation. Defaults to 45.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the current backend is used.

    Returns
    -------
    ZernikeCoefficients
        Zernike standard coefficients in Noll notation.
    """
    return backend.analysis.zernike_standard_coefficients(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
        maximum_term=maximum_term,
    )


@overload
def rms_hoa(
    model: EyeModel | None = None,
    min_order: int = 3,
    max_order: int = 6,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int | None = None,
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> float: ...


@overload
def rms_hoa(
    model: EyeModel | None = None,
    min_order: int = 3,
    max_order: int = 6,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int | None = None,
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[float, Any]: ...


@analysis
def rms_hoa(
    model: EyeModel | None = None,  # noqa: ARG001
    min_order: int = 3,
    max_order: int = 8,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int | None = None,
    *,
    return_raw_result: bool = True,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[float, Any]:
    """Calculates the root-mean-square (RMS) of higher-order aberrations (HOA) in the eye model.

    By default, the Zernike orders from 3 to 8 are used. The RMS is calculated following the definition in the ANSI
    Z80.28-2010 standard [1]_:

    .. math:: RMS_{WFE} = \\sqrt{ \\sum_{n>1, all\\ m} (C_{n}^{m})^2 }

    .. [1]: VC ANSI Z80.28-2010—Ophthalmics—Methods of Reporting Optical Aberrations of Eyes. (n.d.).
            https://webstore.ansi.org/standards/ansi/vcansiz80282010

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the ray trace. If `None`, the current eye model will be used.
    min_order : int
        The minimum Zernike polynomial order to be included in the calculation. Defaults to 3.
    max_order : int
        The maximum Zernike polynomial order to be included in the calculation. Defaults to 8.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate for the Zernike calculation. When `None`, the first field in the backend is used.
        Defaults to `None`.
    wavelength : float | None, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in the backend is used.
        Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    sampling : SampleSize | str | int, optional
        The sampling for the Zernike calculation. Defaults to 64.
    maximum_term : int | None, optional
        The maximum term for the Zernike calculation. If `None`, the maximum term is set to the largest term of
        `max_order`. Defaults to `None`.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the current backend is used.

    Returns
    -------
    float
        The root-mean-square (RMS) of higher-order aberrations (HOA) in the eye model.

    Raises
    ------
    ValueError
        If `min_order` or `max_order` is less than 0.
        If `max_order` is less than or equal to `min_order`.
        If `maximum_term` is less than the largest term of `max_order`.
    """
    if min_order < 0 or max_order < 0:
        raise ValueError("min_order and max_order must be greater than or equal to 0")

    if max_order <= min_order:
        raise ValueError("max_order must be greater than min_order")

    min_index, max_index = min_max_noll_index(min_order, max_order)

    if maximum_term is None:
        maximum_term = max_index
    elif maximum_term < max_index:
        raise ValueError(f"maximum_term must be greater than or equal to the largest term of max_order: {max_index}")

    zernikes, raw_result = backend.analysis.zernike_standard_coefficients(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
        maximum_term=maximum_term,
    )

    rms_aberrations = float(np.sqrt(sum(zernikes[i] ** 2 for i in range(min_index, max_index + 1))))

    return rms_aberrations, raw_result
