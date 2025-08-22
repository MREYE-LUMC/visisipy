from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis

if TYPE_CHECKING:
    from pandas import Series

    from visisipy.backend import BaseBackend
    from visisipy.models import EyeModel
    from visisipy.types import FieldCoordinate, FieldType, SampleSize


@dataclass(frozen=True)
class MTFResult:
    """Result of the MTF analysis.

    Attributes
    ----------
    tangential : Series
        The tangential MTF.
    sagittal : Series
        The sagittal MTF.
    """

    tangential: Series
    """The tangential MTF."""

    sagittal: Series
    """The sagittal MTF."""


@overload
def fft_mtf(
    model: EyeModel | None = None,
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | None = None,
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: Literal[False] = False,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> MTFResult: ...


@overload
def fft_mtf(
    model: EyeModel | None = None,
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | None = None,
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: Literal[True] = True,
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[MTFResult, Any]: ...


@analysis
def fft_mtf(
    model: EyeModel | None = None,  # noqa: ARG001
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | None = None,
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[MTFResult, Any]:
    return backend.analysis.fft_mtf(
        sampling=sampling,
        field_coordinate=field_coordinate,
        field_type=field_type,
        wavelength=wavelength,
        maximum_frequency=maximum_frequency,
    )
