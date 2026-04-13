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
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 128,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: Literal[False] = False,
    backend: BaseBackend = _AUTOMATIC_BACKEND,
) -> MTFResult: ...


@overload
def fft_mtf(
    model: EyeModel | None = None,
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 128,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: Literal[True] = True,
    backend: BaseBackend = _AUTOMATIC_BACKEND,
) -> tuple[MTFResult, Any]: ...


@analysis
def fft_mtf(
    model: EyeModel | None = None,  # noqa: ARG001
    field_coordinate: FieldCoordinate | Literal["all"] = "all",
    field_type: FieldType = "angle",
    wavelength: float | None = None,
    sampling: SampleSize | str | int = 128,
    maximum_frequency: float | Literal["default"] = "default",
    *,
    return_raw_result: bool = False,  # noqa: ARG001
    backend: BaseBackend = _AUTOMATIC_BACKEND,
) -> tuple[MTFResult, Any]:
    """Calculate the FFT Modulation Transfer Function (MTF).

    Parameters
    ----------
    model : EyeModel | None
        The eye model to be used in the analysis. If `None`, the current eye model will be used.
    field_coordinate : FieldCoordinate | Literal["all"]
        The field coordinate(s) at which the MTF is calculated. Can be a specific coordinate (e.g., (0, 0)) or
        "all" to calculate for all fields in the backend. Defaults to "all".
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    wavelength : float | None
        The wavelength at which the MTF is calculated. If `None`, the first wavelength in the backend is used.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where
        N is an integer. Only symmetric sample sizes are supported. Defaults to 128.
    maximum_frequency : float | Literal["default"]
        The maximum frequency (in cycles per millimeter) to calculate the MTF up to. If "default", the default
        maximum frequency is used by the backend. Defaults to "default".
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.
    backend : BaseBackend, optional
        The backend to be used for the analysis. If not provided, the default backend is used.

    Returns
    -------
    MTFResult
        The MTF data as an MTFResult object, which provides access to the tangential and sagittal MTF values for
        each field coordinate.

    Raises
    ------
    ValueError
        If `field_coordinate` is not a valid FieldCoordinate or "all".
    """
    if isinstance(field_coordinate, str) and field_coordinate != "all":
        msg = f"Invalid value for field_coordinate: {field_coordinate}. Expected a FieldCoordinate or 'all'."
        raise ValueError(msg)

    return backend.analysis.fft_mtf(
        field_coordinate=field_coordinate,
        field_type=field_type,
        wavelength=wavelength,
        sampling=sampling,
        maximum_frequency=maximum_frequency,
    )
