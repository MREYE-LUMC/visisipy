from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from types import MethodType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, overload
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from pandas import DataFrame
    from zospy.zpcore import OpticStudioSystem

    from visisipy.analysis.cardinal_points import CardinalPointsResult
    from visisipy.models import BaseEye, EyeModel
    from visisipy.refraction import FourierPowerVectorRefraction
    from visisipy.wavefront import ZernikeCoefficients

_BACKEND: BaseBackend | None = None
_DEFAULT_BACKEND: Backend | str = "opticstudio"


_Analysis = TypeVar("_Analysis", bound=Callable)


class _AnalysisMethod(Generic[_Analysis]):
    def __init__(self, analysis) -> None:
        self._analysis = analysis

    @overload
    def __get__(self, instance: None, owner: type[BaseAnalysisRegistry]) -> _AnalysisMethod[_Analysis]: ...

    @overload
    def __get__(self, instance: BaseAnalysisRegistry, owner: type[BaseAnalysisRegistry]) -> _Analysis: ...

    def __get__(self, instance: BaseAnalysisRegistry | None, owner: type[BaseAnalysisRegistry]) -> _AnalysisMethod | _Analysis:
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
    ) -> CardinalPointsResult: ...

    @abstractmethod
    def raytrace(
        self,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object"] = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> DataFrame: ...

    @abstractmethod
    def refraction(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
    ) -> FourierPowerVectorRefraction: ...

    @abstractmethod
    def zernike_standard_coefficients(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
        field_type: Literal["angle", "object_height"] = "angle",
        sampling: str = "512x512",
        maximum_term: int = 45,
    ) -> ZernikeCoefficients: ...


class BaseBackend(ABC):
    model: BaseEye | None

    @_classproperty
    def analysis(self) -> BaseAnalysisRegistry: ...

    @classmethod
    @abstractmethod
    def build_model(cls, model: EyeModel, **kwargs) -> BaseEye: ...

    @classmethod
    @abstractmethod
    def clear_model(cls) -> None: ...

    @classmethod
    @abstractmethod
    def save_model(cls, filename: str | PathLike | None = None) -> None: ...


class Backend(str, Enum):
    OPTICSTUDIO = "opticstudio"


def set_backend(backend: Backend | str = Backend.OPTICSTUDIO, **kwargs) -> None:
    """Set the backend to use for optical simulations.

    Parameters
    ----------
    backend : Backend | str
        The backend to use. Must be one of the values in the `Backend` enum. Defaults to `Backend.OPTICSTUDIO`.
    kwargs
        Additional keyword arguments to pass to the backend's `initialize` method.

    Raises
    ------
    ValueError
        If an invalid backend is specified.
    """
    global _BACKEND  # noqa: PLW0603

    if _BACKEND is not None:
        warn(
            f"The backend is already set to {_BACKEND.__class__.__name__}. "
            f"Reconfiguring the backend is not recommended and may cause issues."
        )

    if backend == Backend.OPTICSTUDIO:
        os_backend = importlib.import_module("visisipy.opticstudio.backend")
        _BACKEND = os_backend.OpticStudioBackend
        _BACKEND.initialize(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_backend() -> BaseBackend:
    """Get the current backend.

    Returns
    -------
    BaseBackend
        The current backend.
    """
    if _BACKEND is None:
        set_backend(_DEFAULT_BACKEND)

    return _BACKEND


def get_oss() -> OpticStudioSystem | None:
    """
    Get the OpticStudioSystem instance from the current backend.

    Returns
    -------
    OpticStudioSystem | None
        The OpticStudioSystem instance if the current backend is the OpticStudio backend, otherwise `None`.
    """
    os_backend = importlib.import_module("visisipy.opticstudio.backend")

    if _BACKEND is os_backend.OpticStudioBackend:
        return os_backend._oss  # noqa: SLF001

    return None
