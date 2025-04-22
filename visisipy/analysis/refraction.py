"""Calculate the spherical equivalent of refraction of the eye."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis
from visisipy.refraction import FourierPowerVectorRefraction
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from visisipy import EyeModel
    from visisipy.backend import BaseBackend, FieldCoordinate, FieldType
    from visisipy.wavefront import ZernikeCoefficients

__all__ = ("refraction", "zernike_data_to_refraction")


@overload
def refraction(
    model: EyeModel | None = None,
    field_coordinate: tuple[float, float] | None = None,
    sampling: SampleSize | str | int = 64,
    wavelength: float | None = None,
    pupil_diameter: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    *,
    use_higher_order_aberrations: bool = True,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> FourierPowerVectorRefraction: ...


@overload
def refraction(
    model: EyeModel | None = None,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 64,
    pupil_diameter: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    *,
    use_higher_order_aberrations: bool = True,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[FourierPowerVectorRefraction, Any]: ...


@analysis
def refraction(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 64,
    pupil_diameter: float | None = None,
    field_type: FieldType = "angle",
    *,
    use_higher_order_aberrations: bool = True,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[FourierPowerVectorRefraction, Any]:
    """Calculates the ocular refraction.

    The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power vector form.

    Parameters
    ----------
    model : EyeModel, optional
        The eye model to use for the refraction calculation. When `None`, the currently built model is used.
        Defaults to `None`.
    use_higher_order_aberrations : bool, optional
        If `True`, higher-order aberrations are used in the calculation. Defaults to `True`.
    field_coordinate : tuple[float, float], optional
        The field coordinate for the Zernike calculation. When `None`, the default field configured in the backend is
        used. Defaults to `None`.
    wavelength : float, optional
        The wavelength for the Zernike calculation. When `None`, the default wavelength configured in the backend is
        used. Defaults to `None`.
    sampling : SampleSize | str | int, optional
        The sampling for the Zernike calculation. Defaults to 64.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    pupil_diameter : float, optional
        The diameter of the pupil for the refraction calculation. Defaults to the pupil diameter configured in the
        model.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
     FourierPowerVectorRefraction
          The ocular refraction in Fourier power vector form.
    Any
        The raw analysis result from the backend.

    See Also
    --------
    FourierPowerVectorRefraction : Ocular refraction in Fourier power vector form.
    """
    return backend.analysis.refraction(
        use_higher_order_aberrations=use_higher_order_aberrations,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        sampling=SampleSize(sampling),
        pupil_diameter=pupil_diameter,
        field_type=field_type,
    )


def zernike_data_to_refraction(
    zernike_coefficients: ZernikeCoefficients,
    exit_pupil_semi_diameter: float,
    wavelength: float,
    *,
    use_higher_order_aberrations: bool = True,
) -> FourierPowerVectorRefraction:
    """Convert Zernike coefficients to ocular refraction in Fourier power vector form.

    Parameters
    ----------
    zernike_coefficients
    exit_pupil_semi_diameter
    wavelength
    use_higher_order_aberrations

    Returns
    -------
    """
    z4 = zernike_coefficients[4] * wavelength * 4 * np.sqrt(3)
    z11 = zernike_coefficients[11] * wavelength * 12 * np.sqrt(5)
    z22 = zernike_coefficients[22] * wavelength * 24 * np.sqrt(7)
    z37 = zernike_coefficients[37] * wavelength * 40 * np.sqrt(9)

    z6 = zernike_coefficients[6] * wavelength * 2 * np.sqrt(6)
    z12 = zernike_coefficients[12] * wavelength * 6 * np.sqrt(10)
    z24 = zernike_coefficients[24] * wavelength * 12 * np.sqrt(14)
    z38 = zernike_coefficients[38] * wavelength * 60 * np.sqrt(2)

    z5 = zernike_coefficients[5] * wavelength * 2 * np.sqrt(6)
    z13 = zernike_coefficients[13] * wavelength * 6 * np.sqrt(10)
    z23 = zernike_coefficients[23] * wavelength * 12 * np.sqrt(14)
    z39 = zernike_coefficients[39] * wavelength * 60 * np.sqrt(2)

    if use_higher_order_aberrations:
        return FourierPowerVectorRefraction(
            M=(-z4 + z11 - z22 + z37) / (exit_pupil_semi_diameter**2),
            J0=(-z6 + z12 - z24 + z38) / (exit_pupil_semi_diameter**2),
            J45=(-z5 + z13 - z23 + z39) / (exit_pupil_semi_diameter**2),
        )

    return FourierPowerVectorRefraction(
        M=(-z4) / (exit_pupil_semi_diameter**2),
        J0=(-z6) / (exit_pupil_semi_diameter**2),
        J45=(-z5) / (exit_pupil_semi_diameter**2),
    )
