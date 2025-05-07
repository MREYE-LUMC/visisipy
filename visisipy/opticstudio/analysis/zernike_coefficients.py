"""Zernike coefficients analysis for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zospy as zp

from visisipy.types import SampleSize
from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    from zospy.analyses.wavefront.zernike_standard_coefficients import ZernikeStandardCoefficientsResult

    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.opticstudio.backend import OpticStudioBackend


def _build_zernike_result(zernike_result: ZernikeStandardCoefficientsResult, maximum_term: int) -> ZernikeCoefficients:
    return ZernikeCoefficients({k: v.value for k, v in zernike_result.coefficients.items() if k <= maximum_term})


def zernike_standard_coefficients(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
) -> tuple[ZernikeCoefficients, ZernikeStandardCoefficientsResult]:
    """Calculate the Zernike standard coefficients at the retina surface.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
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
    ZernikeCoefficients
        ZOSPy Zernike standard coefficients analysis output.
    """
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = 1 if wavelength is None else backend.get_wavelength_number(wavelength)

    if wavelength_number is None:
        backend.set_wavelengths([wavelength])
        wavelength_number = 1

    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    zernike_result = zp.analyses.wavefront.ZernikeStandardCoefficients(
        sampling=str(sampling),
        maximum_term=maximum_term,
        wavelength=wavelength_number,
        field=1,
        reference_opd_to_vertex=False,
        surface="Image",
        sx=0.0,
        sy=0.0,
        sr=1.0,
    ).run(backend.get_oss())

    return _build_zernike_result(zernike_result.data, maximum_term), zernike_result.data
