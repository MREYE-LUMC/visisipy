"""Zernike coefficients analysis for Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.wavefront import ZernikeOPD

from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import SampleSize
from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.optiland.backend import OptilandBackend


def _build_zernike_coefficients(
    zernike_opd: ZernikeOPD,
) -> ZernikeCoefficients:
    return ZernikeCoefficients(dict(enumerate(zernike_opd.coeffs, start=1)))


def zernike_standard_coefficients(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
) -> tuple[ZernikeCoefficients, ZernikeOPD]:
    """Calculate the Zernike standard coefficients at the retina surface.

    Parameters
    ----------
    backend : type[OptilandBackend]
        Reference to the Optiland backend.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
        Defaults to `None`.
    wavelength : float | None, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
        Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    sampling : SampleSize | str | int, optional
        The sampling for the Zernike calculation. Defaults to 512.
    maximum_term : int, optional
        The maximum term for the Zernike calculation. Defaults to 45.

    Returns
    -------
    AttrDict
        ZOSPy Zernike standard coefficients analysis output.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength = set_wavelength(backend, wavelength)
    normalized_field = set_field(backend, field_coordinate=field_coordinate, field_type=field_type)

    zernike_opd = ZernikeOPD(
        backend.optic,
        field=normalized_field,
        wavelength=wavelength,
        num_rings=int(sampling),
        num_terms=maximum_term,
        zernike_type="noll",
    )

    return _build_zernike_coefficients(zernike_opd), zernike_opd
