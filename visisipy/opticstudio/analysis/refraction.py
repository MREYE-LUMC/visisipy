from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import zospy as zp

from visisipy.opticstudio.analysis.zernike_coefficients import zernike_standard_coefficients
from visisipy.refraction import FourierPowerVectorRefraction

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend


def _get_zernike_coefficient(zernike_result: zp.analyses.base.AttrDict, coefficient: int) -> float:
    return zernike_result.Data.Coefficients.loc["Z" + str(coefficient)].Value


def _zernike_data_to_refraction(
    zernike_data: zp.analyses.base.AttrDict,
    pupil_data: zp.functions.lde.PupilData,
    wavelength: float,
    *,
    use_higher_order_aberrations: bool = True,
) -> FourierPowerVectorRefraction:
    z4 = _get_zernike_coefficient(zernike_data, 4) * wavelength * 4 * np.sqrt(3)
    z11 = _get_zernike_coefficient(zernike_data, 11) * wavelength * 12 * np.sqrt(5)
    z22 = _get_zernike_coefficient(zernike_data, 22) * wavelength * 24 * np.sqrt(7)
    z37 = _get_zernike_coefficient(zernike_data, 37) * wavelength * 40 * np.sqrt(9)

    z6 = _get_zernike_coefficient(zernike_data, 6) * wavelength * 2 * np.sqrt(6)
    z12 = _get_zernike_coefficient(zernike_data, 12) * wavelength * 6 * np.sqrt(10)
    z24 = _get_zernike_coefficient(zernike_data, 24) * wavelength * 12 * np.sqrt(14)
    z38 = _get_zernike_coefficient(zernike_data, 38) * wavelength * 60 * np.sqrt(2)

    z5 = _get_zernike_coefficient(zernike_data, 5) * wavelength * 2 * np.sqrt(6)
    z13 = _get_zernike_coefficient(zernike_data, 13) * wavelength * 6 * np.sqrt(10)
    z23 = _get_zernike_coefficient(zernike_data, 23) * wavelength * 12 * np.sqrt(14)
    z39 = _get_zernike_coefficient(zernike_data, 39) * wavelength * 60 * np.sqrt(2)

    exit_pupil_radius = pupil_data.ExitPupilDiameter / 2

    if use_higher_order_aberrations:
        return FourierPowerVectorRefraction(
            M=(-z4 + z11 - z22 + z37) / (exit_pupil_radius**2),
            J0=(-z6 + z12 - z24 + z38) / (exit_pupil_radius**2),
            J45=(-z5 + z13 - z23 + z39) / (exit_pupil_radius**2),
        )

    return FourierPowerVectorRefraction(
        M=(-z4) / (exit_pupil_radius**2),
        J0=(-z6) / (exit_pupil_radius**2),
        J45=(-z5) / (exit_pupil_radius**2),
    )


def refraction(
    backend: type[OpticStudioBackend],
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    pupil_diameter: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    *,
    use_higher_order_aberrations: bool = True,
) -> tuple[FourierPowerVectorRefraction, zp.analyses.base.AnalysisResult]:
    """Calculates the ocular refraction.

    The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power
    vector form.

    Parameters
    ----------
    use_higher_order_aberrations : bool, optional
        If `True`, higher-order aberrations are used in the calculation. Defaults to `True`.
    field_coordinate : tuple[float, float], optional
        The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
        Defaults to `None`.
    wavelength : float, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
        Defaults to `None`.
    pupil_diameter : float, optional
        The diameter of the pupil for the refraction calculation. Defaults to the pupil diameter configured in the
        model.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".

    Returns
    -------
     FourierPowerVectorRefraction
          The ocular refraction in Fourier power vector form.
    """
    # Get the wavelength from OpticStudio if not specified
    wavelength = backend.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength if wavelength is None else wavelength

    # Temporarily change the pupil diameter
    old_pupil_semi_diameter = None
    if pupil_diameter is not None:
        old_pupil_semi_diameter = backend.model.pupil.semi_diameter
        backend.model.pupil.semi_diameter = pupil_diameter / 2

    pupil_data = zp.functions.lde.get_pupil(backend.oss)
    _, zernike_coefficients = zernike_standard_coefficients(
        backend,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
    )

    if old_pupil_semi_diameter is not None:
        backend.model.pupil.semi_diameter = old_pupil_semi_diameter

    return _zernike_data_to_refraction(
        zernike_coefficients,
        pupil_data,
        wavelength,
        use_higher_order_aberrations=use_higher_order_aberrations,
    ), zernike_standard_coefficients
