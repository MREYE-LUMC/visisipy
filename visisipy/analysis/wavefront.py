"""Calculate wavefront maps of the eye."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy.backend import BaseBackend
    from visisipy.models import EyeModel


@overload
def opd_map(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
    use_exit_pupil_shape: bool = False,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame: ...


@overload
def opd_map(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
    use_exit_pupil_shape: bool = False,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[DataFrame, Any]: ...


@analysis
def opd_map(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
    use_exit_pupil_shape: bool = False,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> DataFrame | tuple[DataFrame, Any]:
    """Calculate the Optical Path Difference (OPD) map at the retina surface.

    Parameters
    ----------
    model : EyeModel | None
        The eye model to use for the wavefront calculation. If `None`, the currently built model will be used.
    field_coordinate : FieldCoordinate | None
        The coordinate of the field for which the wavefront is calculated. If `None`, the current field coordinate will be used.
    wavelength : float | None
        The wavelength (in nm) for which the wavefront is calculated. If `None`, the current wavelength will be used.
    field_type : FieldType
        The type of field coordinate provided. Either 'angle' (degrees) or 'object_height' (mm). Defaults to 'angle'.
    sampling : SampleSize | str | int
        The sampling of the OPD map. Can be an integer (e.g., 128 for 128x128), a string like '128x128', or a `SampleSize` object.
        Defaults to 128.
    remove_tilt : bool, optional
        If `True`, the tilt component is removed from the OPD map. Defaults to `True`.
    use_exit_pupil_shape : bool, optional
        If `True`, the OPD map is distorted to show the shape of the exit pupil. Defaults to `False`. This option is not supported
        by all backends.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : type[BaseBackend]
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the OPD map values in waves.
    Any
        The raw analysis result from the backend.
    """
    return backend.analysis.opd_map(
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        field_type=field_type,
        sampling=SampleSize(sampling),
        remove_tilt=remove_tilt,
        use_exit_pupil_shape=use_exit_pupil_shape,
    )
