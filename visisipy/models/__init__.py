from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import visisipy.backend as _backend
from visisipy.models.geometry import EyeGeometry, NavarroGeometry, create_geometry
from visisipy.models.materials import (
    EyeMaterials,
    NavarroMaterials,
    NavarroMaterials458,
    NavarroMaterials543,
    NavarroMaterials589,
    NavarroMaterials633,
)

if TYPE_CHECKING:
    from os import PathLike

__all__ = (
    "BaseEye",
    "BaseSurface",
    "EyeGeometry",
    "EyeMaterials",
    "EyeModel",
    "NavarroGeometry",
    "NavarroMaterials",
    "NavarroMaterials458",
    "NavarroMaterials543",
    "NavarroMaterials589",
    "NavarroMaterials633",
    "create_geometry",
)


@dataclass
class EyeModel:
    geometry: EyeGeometry = field(default_factory=NavarroGeometry)
    materials: EyeMaterials = field(default_factory=NavarroMaterials)
    _built: BaseEye | None = field(default=None, init=False, repr=False)

    def build(
        self,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ) -> BaseEye:
        backend = _backend.get_backend()
        self._built = backend.build_model(
            self,
            start_from_index=start_from_index,
            replace_existing=replace_existing,
            object_distance=object_distance,
            **kwargs,
        )

        return self._built

    @staticmethod
    def clear() -> None:
        backend = _backend.get_backend()
        backend.clear_model()

    @staticmethod
    def save(filename: str | PathLike | None = None) -> None:
        backend = _backend.get_backend()
        backend.save_model(filename)


class BaseSurface(ABC):
    """Abstract class that must be implemented by backend-specific surface classes."""

    @abstractmethod
    def __init__(
        self,
        comment: str,
        radius: float = float("inf"),
        thickness: float = 0.0,
        semi_diameter: float | None = None,
        conic: float = 0.0,
        material: Any | None = None,
        *,
        is_stop: bool | None = None,
    ): ...

    @property
    @abstractmethod
    def surface(self) -> Any: ...

    @abstractmethod
    def build(self, *args, **kwargs):
        """Build the surface in the backend."""
        ...


class BaseEye(ABC):
    """Abstract class that must be implemented by backend-specific eye model classes."""

    @abstractmethod
    def __init__(self, model: EyeModel): ...

    @abstractmethod
    def build(
        self,
        *args,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ):
        """Build the eye model in the backend."""
        ...

    @property
    @abstractmethod
    def eye_model(self) -> EyeModel: ...

    @property
    def surfaces(self) -> dict[str, BaseSurface]:
        """Dictionary with surface names as keys and surfaces as values."""
        return {k.lstrip("_"): v for k, v in self.__dict__.items() if isinstance(v, BaseSurface)}

    def update_surfaces(self, attribute: str, value: Any, surface_names: list[str] | None = None) -> None:
        """Batch update all surfaces.

        Set `attribute` to `value` for multiple surfaces. If `surfaces` is not specified, all surfaces of the eye
        model are updated.

        Parameters
        ----------
        attribute : str
            Name of the attribute to update
        value : Any
            New value of the surface attribute
        surfaces : list[str]
            List of surfaces to be updated. If not specified, all surfaces are updated.

        Returns
        -------

        """
        surfaces = [self.surfaces[s] for s in surface_names] if surface_names is not None else self.surfaces.values()

        for s in surfaces:
            setattr(s, attribute, value)
