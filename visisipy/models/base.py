"""Base classes for eye models and surfaces."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import visisipy.backend as _backend

if TYPE_CHECKING:
    from os import PathLike

    from visisipy.models.catalog.navarro import NavarroGeometry
    from visisipy.models.geometry import EyeGeometry
    from visisipy.models.materials import EyeMaterials


def _get_default_geometry() -> NavarroGeometry:
    """Get the default eye geometry.

    Returns
    -------
    EyeGeometry
        The default eye geometry.
    """
    # Import here to avoid circular imports
    from visisipy.models.catalog.navarro import NavarroGeometry  # noqa: PLC0415

    return NavarroGeometry()


def _get_default_materials() -> EyeMaterials:
    """Get the default eye materials.

    Returns
    -------
    EyeMaterials
        The default eye materials.
    """
    # Import here to avoid circular imports
    from visisipy.models.materials import NavarroMaterials  # noqa: PLC0415

    return NavarroMaterials()


@dataclass
class EyeModel:
    """Optical model of the eye.

    Visisipy's eye models consist of two parts: the geometry and the material model. The geometry defines the shape of
    the eye, while the material model defines the optical properties of the materials in the eye. By default, this model
    uses the geometry and materials of the Navarro wide-field eye model [1]_. The default material model has been
    fitted to the refractive indices reported by Escudero-Sanz and Navarro [1]_. This model will work with all visible
    wavelengths, but could deviate slightly from the values provided in the literature for the specified wavelengths.

    Attributes
    ----------
    geometry : EyeGeometry
        Geometry of the eye. Defaults to the geometry of the Navarro eye model.
    materials : EyeMaterials
        Properties of the materials of the eye. Defaults to the materials of the Navarro eye model.

    See Also
    --------
    NavarroGeometry : Geometry of the Navarro eye model.
    NavarroMaterials : Materials of the Navarro eye model.

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999).
       Off-axis aberrations of a wide-angle schematic eye model.
       JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881
    """

    geometry: EyeGeometry = field(default_factory=_get_default_geometry)
    materials: EyeMaterials = field(default_factory=_get_default_materials)
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
    def from_dict(cls, data: dict[str, Any]) -> EyeModel:
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
        from visisipy.models.geometry import EyeGeometry  # noqa: PLC0415
        from visisipy.models.materials import EyeMaterials  # noqa: PLC0415

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
        from visisipy import __version__  # noqa: PLC0415

        data = self.to_dict()
        data["visisipy_version"] = __version__
        return json.dumps(data, allow_nan=True, indent=4)

    @classmethod
    def from_json(cls, data: str) -> EyeModel:
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
    def load_json(cls, filename: str | PathLike) -> EyeModel:
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
        """Return ``None`` for this placeholder surface."""
        return None

    def build(self, *args, position: int, **kwargs) -> int:  # noqa: ARG002, PLR6301
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
