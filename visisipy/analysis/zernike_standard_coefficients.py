from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from visisipy.analysis.base import analysis
from visisipy.backend import get_backend

if TYPE_CHECKING:
    from visisipy.models import EyeModel


@analysis
def zernike_standard_coefficients(
    model: EyeModel | None,  # noqa: ARG001
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    sampling: str = "512x512",
    maximum_term: int = 45,
    *,
    return_raw_result: bool = False,  # noqa: ARG001
) -> tuple[dict, Any]:
    """
    Calculates the Zernike standard coefficients at the retina surface.

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
    sampling : str, optional
        The sampling for the Zernike calculation. Defaults to "512x512".
    maximum_term : int, optional
        The maximum term for the Zernike calculation. Defaults to 45.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.

    Returns
    -------
    AttrDict
        ZOSPy Zernike standard coefficients analysis output.
    """
    return get_backend().analysis.zernike_standard_coefficients(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=sampling,
        maximum_term=maximum_term,
    )
