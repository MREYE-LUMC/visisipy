"""Refraction analysis for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

import zospy as zp

from visisipy.analysis.refraction import zernike_data_to_refraction
from visisipy.opticstudio.analysis.zernike_coefficients import (
    zernike_standard_coefficients,
)
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from zospy.analyses.wavefront.zernike_standard_coefficients import ZernikeStandardCoefficientsResult

    from visisipy.opticstudio.backend import OpticStudioBackend
    from visisipy.refraction import FourierPowerVectorRefraction
    from visisipy.types import FieldCoordinate, FieldType


def refraction(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 64,
    pupil_diameter: float | None = None,
    field_type: FieldType = "angle",
    *,
    use_higher_order_aberrations: bool = True,
) -> tuple[FourierPowerVectorRefraction, ZernikeStandardCoefficientsResult]:
    """Calculates the ocular refraction.

    The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power
    vector form.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
        Defaults to `None`.
    wavelength : float, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
        Defaults to `None`.
    sampling : SampleSize | str | int, optional
        The sampling for the Zernike calculation. Defaults to 64.
    pupil_diameter : float, optional
        The diameter of the pupil for the refraction calculation. Defaults to the pupil diameter configured in the
        backend. If the aperture type is "float_by_stop_size", the value is interpreted as the pupil diameter.
        For other aperture types, it is interpreted as the aperture value.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    use_higher_order_aberrations : bool, optional
        If `True`, higher-order aberrations are used in the calculation. Defaults to `True`.

    Returns
    -------
     FourierPowerVectorRefraction
          The ocular refraction in Fourier power vector form.
    """
    # Get the wavelength from OpticStudio if not specified
    wavelength = backend.get_wavelengths()[0] if wavelength is None else wavelength

    # Temporarily change the pupil diameter
    old_pupil_value = None
    if pupil_diameter is not None:
        if backend.get_setting("aperture_type") != "float_by_stop_size":
            message = (
                "When updating the pupil size for aperture types other than 'float_by_stop_size', "
                "the pupil_diameter parameter changes the aperture value instead of the pupil size."
            )
            warn(message, UserWarning, stacklevel=2)

        _, old_pupil_value = backend.get_aperture()

        backend.update_pupil(pupil_diameter)

    pupil_data = zp.functions.lde.get_pupil(backend.get_oss())
    zernike_coefficients, raw_result = zernike_standard_coefficients(
        backend,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
    )

    # Restore the original pupil diameter
    if old_pupil_value is not None:
        backend.update_pupil(old_pupil_value)

    return zernike_data_to_refraction(
        zernike_coefficients,
        pupil_data.ExitPupilDiameter / 2,
        wavelength,
        use_higher_order_aberrations=use_higher_order_aberrations,
    ), raw_result
