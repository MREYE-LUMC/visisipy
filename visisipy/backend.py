from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Literal
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from pandas import DataFrame
    from zospy.zpcore import OpticStudioSystem

    from visisipy.models import BaseEye, EyeModel
    from visisipy.refraction import FourierPowerVectorRefraction

_BACKEND: BaseBackend | None = None
_DEFAULT_BACKEND: Backend | str = "opticstudio"


class _classproperty(property):  # noqa: N801
    def __get__(self, instance, owner=None):
        return self.fget(owner)


class BaseAnalysis(ABC):
    @staticmethod
    @abstractmethod
    def raytrace(
        self,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object"] = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> DataFrame: ...

    @staticmethod
    @abstractmethod
    def refraction(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
    ) -> FourierPowerVectorRefraction: ...


class BaseBackend(ABC):
    model: BaseEye | None

    @_classproperty
    def analysis(self) -> BaseAnalysis: ...

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

    if _BACKEND is os_backend.OpicStudioBackend:
        return os_backend._oss  # noqa: SLF001

    return None
