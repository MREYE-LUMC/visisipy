from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from os import PathLike
from typing import TYPE_CHECKING, Literal
from warnings import warn

if TYPE_CHECKING:
    from visisipy.models import BaseEye, EyeModel

_BACKEND: BaseBackend | None = None
_DEFAULT_BACKEND: Literal["opticstudio"] = "opticstudio"


class BaseBackend(ABC):
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
