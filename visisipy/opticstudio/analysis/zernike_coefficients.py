from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import zospy as zp

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend


def zernike_standard_coefficients(
    backend: type[OpticStudioBackend],
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    sampling: str = "512x512",
    maximum_term: int = 45,
) -> tuple[zp.analyses.base.AttrDict, zp.analyses.base.AnalysisResult]:
    """
    Calculates the Zernike standard coefficients at the retina surface.

    Parameters
    ----------
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
        Defaults to `None`.
    wavelength : float | None, optional
        The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
        Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    sampling : str, optional
        The sampling for the Zernike calculation. Defaults to "512x512".
    maximum_term : int, optional
        The maximum term for the Zernike calculation. Defaults to 45.

    Returns
    -------
    AttrDict
        ZOSPy Zernike standard coefficients analysis output.
    """
    wavelength_number = 1 if wavelength is None else backend.get_wavelength_number(wavelength)

    if wavelength_number is None:
        backend.set_wavelengths([wavelength])
        wavelength_number = 1

    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    zernike_result = zp.analyses.wavefront.zernike_standard_coefficients(
        backend.oss,
        sampling=sampling,
        maximum_term=maximum_term,
        wavelength=wavelength_number,
        field=1,
        reference_opd_to_vertex=False,
        surface="Image",
        sx=0.0,
        sy=0.0,
        sr=1.0,
    )

    return zernike_result.Data, zernike_result
