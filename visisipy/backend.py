from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
from warnings import warn

if sys.version_info <= (3, 11):
    from typing_extensions import NotRequired, TypedDict, Unpack
else:
    from typing import NotRequired, TypedDict, Unpack

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from optiland.optic import Optic
    from pandas import DataFrame
    from zospy.zpcore import OpticStudioSystem

    from visisipy.analysis.cardinal_points import CardinalPointsResult
    from visisipy.models import BaseEye, EyeModel
    from visisipy.refraction import FourierPowerVectorRefraction
    from visisipy.types import SampleSize
    from visisipy.wavefront import ZernikeCoefficients

__all__ = ("Backend", "set_backend", "get_backend", "get_oss", "TypedDict", "Unpack")


_BACKEND: type[BaseBackend] | None = None
_DEFAULT_BACKEND: Backend | Literal["opticstudio", "optiland"] = "opticstudio"


_Analysis = TypeVar("_Analysis", bound=Callable)


class _AnalysisMethod(Generic[_Analysis]):
    def __init__(self, analysis) -> None:
        self._analysis = analysis

    @overload
    def __get__(
        self, instance: None, owner: type[BaseAnalysisRegistry]
    ) -> _AnalysisMethod[_Analysis]: ...

    @overload
    def __get__(
        self, instance: BaseAnalysisRegistry, owner: type[BaseAnalysisRegistry]
    ) -> _Analysis: ...

    def __get__(
        self, instance: BaseAnalysisRegistry | None, owner: type[BaseAnalysisRegistry]
    ) -> _AnalysisMethod | _Analysis:
        if instance is None:
            return self

        return MethodType(self._analysis, instance.backend)


class _classproperty(property):  # noqa: N801
    def __get__(self, instance, owner=None):
        return self.fget(owner)


class BaseAnalysisRegistry(ABC):
    def __init__(self, backend: BaseBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> BaseBackend:
        return self._backend

    @abstractmethod
    def cardinal_points(
        self,
        surface_1: int | None = None,
        surface_2: int | None = None,
    ) -> tuple[CardinalPointsResult, Any]: ...

    @abstractmethod
    def raytrace(
        self,
        coordinates: Iterable[FieldCoordinate] | None = None,
        wavelengths: Iterable[float] | None = None,
        field_type: FieldType = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> tuple[DataFrame, Any]: ...

    @abstractmethod
    def refraction(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        sampling: SampleSize | str | int = 64,
        pupil_diameter: float | None = None,
        field_type: FieldType = "angle",
        *,
        use_higher_order_aberrations: bool = True,
    ) -> tuple[FourierPowerVectorRefraction, Any]: ...

    @abstractmethod
    def zernike_standard_coefficients(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        sampling: SampleSize | str | int = 64,
        maximum_term: int = 45,
    ) -> tuple[ZernikeCoefficients, Any]: ...


ApertureType = Literal[
    "float_by_stop_size",
    "entrance_pupil_diameter",
    "image_f_number",
    "object_numeric_aperture",
]
FieldType = Literal["angle", "object_height"]
FieldCoordinate = tuple[float, float]


class BackendSettings(TypedDict, total=False):
    """A dictionary containing the settings for the backend."""

    field_type: FieldType
    """The field type to use in the optical system. Must be one of 'angle' or 'object_height'."""

    fields: list[FieldCoordinate]
    """List of field coordinates to use in the optical system."""

    wavelengths: list[float]
    """List of wavelengths to use in the optical system."""

    aperture_type: ApertureType
    """
    The aperture type to use in the optical system. Must be one of 'float_by_stop_size', 'entrance_pupil_diameter',
    'image_f_number', or 'object_numeric_aperture'.
    """

    aperture_value: NotRequired[float]
    """The aperture value to use in the optical system. Not required for 'float_by_stop_size'."""


class BaseBackend(ABC):
    model: BaseEye | None
    settings: BackendSettings

    analysis: ClassVar[BaseAnalysisRegistry]

    @classmethod
    @abstractmethod
    def update_settings(cls, **settings: Unpack[BackendSettings]) -> None: ...

    @classmethod
    @abstractmethod
    def build_model(cls, model: EyeModel, **kwargs) -> BaseEye: ...

    @classmethod
    @abstractmethod
    def clear_model(cls) -> None: ...

    @classmethod
    @abstractmethod
    def save_model(cls, filename: str | PathLike | None = None) -> None: ...

    @classmethod
    def save_settings(cls, filename: str | PathLike) -> None:
        if not str(filename).endswith(".json"):
            raise ValueError("Settings file must have a '.json' extension.")

        Path(filename).write_text(json.dumps(cls.settings, indent=4, sort_keys=True))


class Backend(str, Enum):
    OPTICSTUDIO = "opticstudio"
    OPTILAND = "optiland"


def set_backend(
    backend: Backend | Literal["opticstudio", "optiland"] = Backend.OPTICSTUDIO,
    *,
    settings: BackendSettings | None = None,
) -> None:
    """Set the backend to use for optical simulations.

    Parameters
    ----------
    backend : Backend | str
        The backend to use. Must be one of the values in the `Backend` enum. Defaults to `Backend.OPTICSTUDIO`.
    settings : BackendSettings, optional
        Dictionary with settings for the backend. Defaults to `None`.

    Raises
    ------
    ValueError
        If an invalid backend is specified.
    """
    settings = settings or {}

    global _BACKEND  # noqa: PLW0603

    if _BACKEND is not None:
        warn(
            f"The backend is already set to {_BACKEND.__name__}. "
            f"Reconfiguring the backend is not recommended and may cause issues."
        )

    if backend == Backend.OPTICSTUDIO:
        from visisipy.opticstudio import OpticStudioBackend

        _BACKEND = OpticStudioBackend
        _BACKEND.initialize(settings=settings)
    elif backend == Backend.OPTILAND:
        from visisipy.optiland import OptilandBackend

        _BACKEND = OptilandBackend
        _BACKEND.initialize(settings=settings)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_backend() -> type[BaseBackend]:
    """Get the current backend.

    The backend is set to the default backend if it has not been set yet.

    Returns
    -------
    BaseBackend
        The current backend.
    """
    if _BACKEND is None:
        set_backend(_DEFAULT_BACKEND)

    return cast(type[BaseBackend], _BACKEND)


def get_oss() -> OpticStudioSystem | None:
    """
    Get the OpticStudioSystem instance from the current backend.

    Returns
    -------
    OpticStudioSystem | None
        The OpticStudioSystem instance if the current backend is the OpticStudio backend, otherwise `None`.
    """
    from visisipy.opticstudio import OpticStudioBackend

    if _BACKEND is OpticStudioBackend:
        return OpticStudioBackend.oss

    return None


def get_optic() -> Optic | None:
    """
    Get the Optic instance from the current backend.

    Returns
    -------
    Optic
        The Optic instance if the current backend is the Optiland backend, otherwise `None`.
    """
    from visisipy.optiland import OptilandBackend

    if _BACKEND is OptilandBackend:
        return OptilandBackend.optic

    return None
