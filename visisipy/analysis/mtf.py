"""Calculate the modulation transfer function (MTF) of an eye model."""

from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

from visisipy.analysis.base import _AUTOMATIC_BACKEND, analysis
from visisipy.types import FieldCoordinate

if TYPE_CHECKING:
    from pandas import Series

    from visisipy.backend import BaseBackend
    from visisipy.models import EyeModel
    from visisipy.types import FieldType, SampleSize

__all__ = ("MTFResult", "SingleMTFResult", "fft_mtf")


@dataclass(frozen=True)
class SingleMTFResult:
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


class MTFResult(UserDict[FieldCoordinate, SingleMTFResult]):
    """Result of the MTF analysis for multiple fields."""

    def _get_single_field(self) -> SingleMTFResult:
        if len(self) != 1:
            raise ValueError("Multiple fields present. Please specify a field coordinate.")

        return next(iter(self.values()))

    def tangential(self, field: FieldCoordinate | None = None) -> Series:
        """Get the tangential MTF for a specific field.

        Parameters
        ----------
        field : FieldCoordinate, optional
            The field coordinate for which to get the tangential MTF.
            If not specified, the tangential MTF for the single field will be returned.
            If multiple fields are present, a ValueError will be raised.

        Returns
        -------
        Series
            The tangential MTF for the specified field.
        """
        if field is None:
            return self._get_single_field().tangential

        return self[field].tangential

    def sagittal(self, field: FieldCoordinate | None = None) -> Series:
        """Get the sagittal MTF for a specific field.

        Parameters
        ----------
        field : FieldCoordinate, optional
            The field coordinate for which to get the sagittal MTF.
            If not specified, the sagittal MTF for the single field will be returned.
            If multiple fields are present, a ValueError will be raised.

        Returns
        -------
        Series
            The sagittal MTF for the specified field.
        """
        if field is None:
            return self._get_single_field().sagittal

        return self[field].sagittal


@overload
def fft_mtf(
    model: EyeModel | None = None,
    sampling: SampleSize | str | int = 64,
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
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
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
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
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: type[BaseBackend] = _AUTOMATIC_BACKEND,
) -> tuple[MTFResult, Any]:
    if isinstance(field_coordinate, str) and field_coordinate != "all":
        msg = f"Invalid value for field_coordinate: {field_coordinate}. Expected a FieldCoordinate or 'all'."
        raise ValueError(msg)

    return backend.analysis.fft_mtf(
        sampling=sampling,
        field_coordinate=field_coordinate,
        field_type=field_type,
        wavelength=wavelength,
        maximum_frequency=maximum_frequency,
    )
