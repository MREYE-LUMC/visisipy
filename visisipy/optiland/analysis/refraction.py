"""Spherical equivalent of refraction analysis for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    old_aperture = None
    if pupil_diameter is not None:
        old_aperture = backend.get_optic().aperture
        backend.model.pupil.surface.set_semi_aperture(pupil_diameter / 2)

    zernike_coefficients, zernike_opd = zernike_standard_coefficients(
        backend=backend,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
    )

    exit_pupil_semi_diameter = backend.get_optic().paraxial.XPD() / 2

    if old_aperture is not None:
        backend.get_optic().set_aperture(aperture_type=old_aperture.ap_type, value=old_aperture.value)

    if wavelength is None:
        wavelength = backend.get_wavelengths()[0]

    fourier_refraction = zernike_data_to_refraction(
        zernike_coefficients,
        exit_pupil_semi_diameter=exit_pupil_semi_diameter,
        wavelength=wavelength,
        use_higher_order_aberrations=use_higher_order_aberrations,
    )

    return fourier_refraction, zernike_opd
