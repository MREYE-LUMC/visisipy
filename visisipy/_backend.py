from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from os import PathLike
from typing import Iterable, TYPE_CHECKING, Literal
from warnings import warn

if TYPE_CHECKING:
    from pandas import DataFrame
    from visisipy.models import BaseEye, EyeModel
    from visisipy.analysis import FourierPowerVectorRefraction

_BACKEND: BaseBackend | None = None
_DEFAULT_BACKEND: Literal["opticstudio"] = "opticstudio"


class _classproperty(property):
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
    ) -> DataFrame:
        ...

    @staticmethod
    @abstractmethod
    def refraction(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
    ) -> FourierPowerVectorRefraction:
        ...


class BaseBackend(ABC):
    _model: BaseEye | None

    @_classproperty
    def analysis(self) -> BaseAnalysis:
        ...

    @classmethod
    @abstractmethod
    def build_model(cls, model: EyeModel, **kwargs) -> BaseEye:
        ...

    @classmethod
    @abstractmethod
    def clear_model(cls) -> None:
        ...

    @classmethod
    @abstractmethod
    def save_model(cls, filename: str | PathLike | None = None) -> None:
        ...


def set_backend(backend: Literal["opticstudio"] = "opticstudio", **kwargs) -> None:
    global _BACKEND

    if _BACKEND is not None:
        warn(
            f"The backend is already set to {_BACKEND.__class__.__name__}. "
            f"Reconfiguring the backend is not recommended and may cause issues."
        )

    if backend == "opticstudio":
        os_backend = importlib.import_module("visisipy.opticstudio.backend")
        _BACKEND = os_backend.OpticStudioBackend
        _BACKEND.initialize(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_backend() -> BaseBackend:
    global _BACKEND

    if _BACKEND is None:
        set_backend(_DEFAULT_BACKEND)

    return _BACKEND