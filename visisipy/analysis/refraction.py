from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from visisipy.analysis.base import analysis

if TYPE_CHECKING:
    from visisipy import EyeModel
    from visisipy.backend import BaseBackend
    from visisipy.refraction import FourierPowerVectorRefraction

__all__ = ("refraction",)


@overload
def refraction(
    model: EyeModel | None,
    field_coordinate: tuple[float, float] | None,
    wavelength: float | None,
    pupil_diameter: float | None,
    field_type: Literal["angle", "object_height"],
    *,
    use_higher_order_aberrations: bool,
    return_raw_result: Literal[False],
    backend: type[BaseBackend],
) -> FourierPowerVectorRefraction: ...


@overload
def refraction(
    model: EyeModel | None,
    field_coordinate: tuple[float, float] | None,
    wavelength: float | None,
    pupil_diameter: float | None,
    field_type: Literal["angle", "object_height"],
    *,
    use_higher_order_aberrations: bool,
    return_raw_result: Literal[True],
    backend: type[BaseBackend],
) -> tuple[FourierPowerVectorRefraction, Any]: ...


@analysis
def refraction(
    model: EyeModel | None,  # noqa: ARG001
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    pupil_diameter: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    *,
    use_higher_order_aberrations: bool = True,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend],
) -> FourierPowerVectorRefraction | tuple[FourierPowerVectorRefraction, Any]:
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
    """
    return backend.analysis.refraction(
        use_higher_order_aberrations=use_higher_order_aberrations,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        pupil_diameter=pupil_diameter,
        field_type=field_type,
    )
