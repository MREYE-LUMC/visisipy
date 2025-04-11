from __future__ import annotations

from typing import TYPE_CHECKING

import zospy as zp

from visisipy.types import SampleSize
from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    from visisipy.backend import FieldCoordinate, FieldType
    from visisipy.opticstudio.backend import OpticStudioBackend


def _get_zernike_coefficient(zernike_result: zp.analyses.base.AttrDict, coefficient: int) -> float:
    return zernike_result.Data.Coefficients.loc["Z" + str(coefficient)].Value


def _build_zernike_result(zernike_result: zp.analyses.base.AttrDict, maximum_term: int) -> ZernikeCoefficients:
    return ZernikeCoefficients({i: _get_zernike_coefficient(zernike_result, i) for i in range(1, maximum_term + 1)})


def zernike_standard_coefficients(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    maximum_term: int = 45,
) -> tuple[ZernikeCoefficients, zp.analyses.base.AnalysisResult]:
    """
    Calculate the Zernike standard coefficients at the retina surface.

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
    AttrDict
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

    zernike_result = zp.analyses.wavefront.zernike_standard_coefficients(
        backend.oss,
        sampling=str(sampling),
        maximum_term=maximum_term,
        wavelength=wavelength_number,
        field=1,
        reference_opd_to_vertex=False,
        surface="Image",
        sx=0.0,
        sy=0.0,
        sr=1.0,
    )

    return _build_zernike_result(zernike_result, maximum_term), zernike_result
