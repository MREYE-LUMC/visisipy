"""Base classes for eye models and surfaces."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import visisipy.backend as _backend
from visisipy.types import TypeVar

if TYPE_CHECKING:
    from os import PathLike
    from typing import Self

    from visisipy.models.geometry import EyeGeometry, Surface  # ruff: ignore[unused-import]
    from visisipy.models.materials import EyeMaterials  # ruff: ignore[unused-import]

_Geometry = TypeVar(
    "_Geometry",
    bound="EyeGeometry[Surface, Surface, Surface, Surface, Surface, Surface]",
    "_Geometry",
    bound="EyeGeometry[Surface, Surface, Surface, Surface, Surface, Surface]",
)
_Materials = TypeVar("_Materials", bound="EyeMaterials")


@dataclass
class EyeModel(Generic[_Geometry, _Materials]):
    """Optical model of the eye.

    Visisipy's eye models consist of two parts: the geometry and the material model. The geometry defines the shape of
    the eye, while the material model defines the optical properties of the materials in the eye. Use a concrete
    catalog model such as :class:`NavarroEyeModel` when a predefined model is desired.
    the eye, while the material model defines the optical properties of the materials in the eye. Use a concrete
    catalog model such as :class:`NavarroEyeModel` when a predefined model is desired.

    Attributes
    ----------
    geometry : EyeGeometry
        Explicit geometry specification for the eye.
        Explicit geometry specification for the eye.
    materials : EyeMaterials
        Explicit material specification for the eye.
        Explicit material specification for the eye.

    See Also
    --------
    NavarroEyeModel : Predefined Navarro eye model.
    NavarroEyeModel : Predefined Navarro eye model.
    """

    geometry: _Geometry
    materials: _Materials
    _built: BaseEye | None = field(default=None, init=False, repr=False)

    def build(
        self,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ) -> BaseEye:
        """Build the eye model in the backend.

        If no backend has yet been initialized, the default backend is used. The eye model is built in the backend
        starting from the surface at index `start_from_index`. The cornea front surface will be located at
        `start_from_index + 1`. If `replace_existing` is set to `True`, the existing model in the backend will be
        overwritten. The `object_distance` parameter is used to set the distance between the cornea front and the
        surface preceding the eye model.

        Parameters
        ----------
        start_from_index : int
            Index of the surface after which the eye model will be built.
        replace_existing : bool
            If `True`, the existing model in the backend will be overwritten.
        object_distance : float
            Distance between the cornea front and the surface preceding the eye model.
        **kwargs
            Additional keyword arguments to be passed to the backend when building the model.

        Returns
        -------
        BaseEye
            The built eye model in the backend.
        """
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
        """Clear the model in the backend."""
        backend = _backend.get_backend()
        backend.clear_model()

    @staticmethod
    def save(filename: str | PathLike | None = None) -> None:
        """Save the model definition to a file.

        Saving is done by the backend. As a consequence, the file format depends on the backend used.

        Parameters
        ----------
        filename : str | PathLike | None
            Name of the file to save the model. If `None`, the model will be saved in the default location.
            The file format depends on the backend used.
        """
        backend = _backend.get_backend()
        backend.save_model(filename)

    def to_dict(self) -> dict[str, Any]:
        """Convert the eye model to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the eye model with serialized ``geometry`` and ``materials``.
        """
        return {
            "geometry": self.geometry.to_dict(),
            "materials": self.materials.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an eye model from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary with the eye model parameters, as produced by :meth:`to_dict`.

        Returns
        -------
        EyeModel
            An eye model instance with geometry and materials reconstructed from ``data``.
        """
        # ruff: ignore[import-outside-top-level]
        from visisipy.models.geometry import EyeGeometry

        # ruff: ignore[import-outside-top-level]
        from visisipy.models.materials import EyeMaterials

        return cls(
            geometry=EyeGeometry.from_dict(data["geometry"]),
            materials=EyeMaterials.from_dict(data["materials"]),
        )

    def to_json(self) -> str:
        """Serialize the eye model to a JSON string.

        Returns
        -------
        str
            A JSON representation of the eye model.
        """
        from visisipy import __version__  # ruff: ignore[import-outside-top-level]

        data = self.to_dict()
        data["visisipy_version"] = __version__
        return json.dumps(data, allow_nan=True, indent=4)

    @classmethod
    def from_json(cls, data: str) -> Self:
        """Create an eye model from a JSON string.

        Parameters
        ----------
        data : str
            A JSON string representing an eye model.

        Returns
        -------
        EyeModel
            An eye model instance reconstructed from ``data``.

        Raises
        ------
        ValueError
            If the JSON does not contain a ``"visisipy_version"`` key.
        """
        parsed = json.loads(data)
        if "visisipy_version" not in parsed:
            msg = "JSON data is missing required 'visisipy_version' key. Ensure the data was created with EyeModel.to_json()."
            raise ValueError(msg)

        parsed.pop("visisipy_version")
        return cls.from_dict(parsed)

    def save_json(self, filename: str | PathLike) -> None:
        """Save the eye model as JSON.

        Parameters
        ----------
        filename : str | PathLike
            Path to the output JSON file.

        Raises
        ------
        ValueError
            If the output filename does not have a .json extension.
        """
        filename = Path(filename)

        if not filename.suffix == ".json":
            raise ValueError("Output filename must have a .json extension.")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, filename: str | PathLike) -> Self:
        """Load an eye model from a JSON file.

        Parameters
        ----------
        filename : str | PathLike
            Path to the input JSON file.

        Returns
        -------
        EyeModel
            An eye model instance reconstructed from the JSON file.
        """
        with open(filename, encoding="utf-8") as f:
            return cls.from_json(f.read())


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
    def surface(self) -> Any:
        """Backend-native surface object."""

    @abstractmethod
    def build(self, *args, position: int, replace_existing: bool = False) -> int:
        """Build the surface in the backend."""


class NoSurface(BaseSurface):
    """Dummy surface class for when no surface is needed.

    This is a generic implementation that works with all backends, because it does not modify the optical system.
    """

    def __init__(self, *args, **kwargs):
        pass

    @property
    def surface(self) -> None:
        """Surface is always `None` for this surface type."""
        return None

    def build(self, *args, position: int, **kwargs) -> int:  # ruff: ignore[unused-method-argument]
        """Advance the build index without adding a physical surface.

        Returns
        -------
        int
            The previous surface position.
        """
        return position - 1


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
        """Build the eye model in the backend.

        If no backend has yet been initialized, the default backend is used. The eye model is built in the backend
        starting from the surface at index `start_from_index`. The cornea front surface will be located at
        `start_from_index + 1`. If `replace_existing` is set to `True`, the existing model in the backend will be
        overwritten. The `object_distance` parameter is used to set the distance between the cornea front and the
        surface preceding the eye model.

        Parameters
        ----------
        start_from_index : int
            Index of the surface after which the eye model will be built.
        replace_existing : bool
            If `True`, the existing model in the backend will be overwritten.
        object_distance : float
            Distance between the cornea front and the surface preceding the eye model.
        args
            Additional positional arguments to be passed to the backend when building the model.
        kwargs
            Additional keyword arguments to be passed to the backend when building the model.
        """

    @property
    @abstractmethod
    def eye_model(self) -> EyeModel:
        """Eye-model specification from which this backend-specific eye was built."""

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
        surface_names : list[str]
            List of surfaces to be updated. If not specified, all surfaces are updated.
        """
        surfaces = [self.surfaces[s] for s in surface_names] if surface_names is not None else self.surfaces.values()

        for s in surfaces:
            setattr(s, attribute, value)
