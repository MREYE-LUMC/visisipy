"""Spherical equivalent of refraction analysis for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from visisipy.analysis.refraction import zernike_data_to_refraction
from visisipy.optiland.analysis.zernike_coefficients import (
    zernike_standard_coefficients,
)
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from optiland.wavefront import ZernikeOPD

    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.optiland.backend import OptilandBackend
    from visisipy.refraction import FourierPowerVectorRefraction

__all__ = ("refraction",)


def refraction(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 64,
    pupil_diameter: float | None = None,
    field_type: FieldType = "angle",
    *,
    use_higher_order_aberrations: bool = True,
) -> tuple[FourierPowerVectorRefraction, ZernikeOPD]:
    """Calculates the ocular refraction.

    The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power
    vector form.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the OptilandBackend backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate for the Zernike calculation. When `None`, the first field in Optiland is used.
        Defaults to `None`.
    wavelength : float, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in Optiland is used.
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
    old_aperture = None
    if pupil_diameter is not None:
        if backend.get_setting("aperture_type") != "float_by_stop_size":
            message = (
                "When updating the pupil size for aperture types other than 'float_by_stop_size', "
                "the pupil_diameter parameter changes the aperture value instead of the pupil size."
            )
            warn(message, UserWarning, stacklevel=2)

        old_aperture = backend.get_optic().aperture
        backend.update_pupil(pupil_diameter)

    zernike_coefficients, zernike_opd = zernike_standard_coefficients(
        backend=backend,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
    )

    exit_pupil_semi_diameter = backend.get_optic().paraxial.XPD() / 2

    if old_aperture is not None:
        backend.update_pupil(old_aperture.value)

    if wavelength is None:
        wavelength = backend.get_wavelengths()[0]

    fourier_refraction = zernike_data_to_refraction(
        zernike_coefficients,
        exit_pupil_semi_diameter=exit_pupil_semi_diameter,
        wavelength=wavelength,
        use_higher_order_aberrations=use_higher_order_aberrations,
    )

    return fourier_refraction, zernike_opd
