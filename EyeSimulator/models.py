from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Any

import zospy as zp
import zospy.api._ZOSAPI as _ZOSAPI
from zospy.zpcore import OpticStudioSystem


class Surface:
    def __init__(
        self,
        comment: str,
        radius: float = float("inf"),
        thickness: float = 0.0,
        semi_diameter: float = None,
        conic: float = 0.0,
        refractive_index: float = None,
        is_stop: bool = None,
    ):
        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._refractive_index = refractive_index
        self._is_stop = is_stop

        self._surface = None
        self._is_built = False

    @property
    def comment(self) -> str:
        """Surface description."""
        return self._get_surface_property("Comment")

    @comment.setter
    def comment(self, value: str) -> None:
        self._set_surface_property("Comment", value)

    @property
    def radius(self) -> float:
        """Radius of curvature, in mm."""
        return self._get_surface_property("Radius")

    @radius.setter
    def radius(self, value: float) -> None:
        self._set_surface_property("Radius", value)

    @property
    def thickness(self) -> float:
        """Distance to the next surface, in mm."""
        return self._get_surface_property("Thickness")

    @thickness.setter
    def thickness(self, value: float) -> None:
        self._set_surface_property("Thickness", value)

    @property
    def semi_diameter(self) -> float:
        """Semi diameter (radius) of the pupil at this surface, in mm."""
        return self._get_surface_property("SemiDiameter")

    @semi_diameter.setter
    def semi_diameter(self, value: float) -> None:
        self._set_surface_property("SemiDiameter", value)

    @property
    def conic(self) -> float:
        """Asphericity of the surface, as conic constant.

        For more information about the definition of the conic constant, see
        https://en.wikipedia.org/wiki/Conic_constant.
        """
        return self._get_surface_property("Conic")

    @conic.setter
    def conic(self, value: float) -> None:
        self._set_surface_property("Conic", value)

    @property
    def refractive_index(self):
        """Refractive index of the medium."""
        if self._is_built:
            if (m := self.surface.MaterialCell.GetSolveData()._S_MaterialModel) is not None:
                return float(m.IndexNd)

            return None

        return self._refractive_index

    @refractive_index.setter
    def refractive_index(self, value: float) -> None:
        if self._is_built:
            zp.solvers.material_model(self.surface.MaterialCell, refractive_index=value)
        else:
            self._refractive_index = value

    @property
    def is_stop(self) -> bool:
        """`True` if this is a STOP surface."""
        return self._get_surface_property("IsStop")

    @is_stop.setter
    def is_stop(self, value: bool) -> None:
        self._set_surface_property("IsStop", value)

    @property
    def surface(self) -> _ZOSAPI.Editors.LDE.ILDERow | None:
        """OpticStudio surface object.

        This property only has a value if the surface has been built with `Surface.build`.
        """
        return self._surface

    def build(self, oss: OpticStudioSystem, position: int, replace_existing: bool = False):
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` `oss` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the surface is created.
        position : int
            Index at which the surface is located.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.
        """
        self._surface = oss.LDE.GetSurfaceAt(position) if replace_existing else oss.LDE.InsertNewSurfaceAt(position)

        self.surface.Comment = self.comment
        self.surface.Radius = self.radius
        self.surface.Thickness = self.thickness
        self.surface.Conic = self.conic

        if self.semi_diameter is not None:
            self.surface.SemiDiameter = self.semi_diameter

        if self.refractive_index is not None:
            zp.solvers.material_model(self.surface.MaterialCell, refractive_index=self.refractive_index)

        # Only set IsStop when explicitly specified
        if self.is_stop is not None:
            self.surface.IsStop = self.is_stop

        self._is_built = True

    def relink_surface(self, oss: OpticStudioSystem) -> bool:
        """Link an OpticStudio surface based on its comment.

        Searches for a surface in `oss` whose comment matches `self.comment`.
        This surface is assigned to `self.surface`. If no surface is found or multiple
        surfaces are found, `self.surface` is not updated.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the eye model is defined.

        Returns
        -------
        bool
            True if the operation succeeded, False if the system has not been built, or
            no / multiple surfaces have been found.
        """
        if self._is_built:
            surfaces = zp.functions.lde.find_surface_by_comment(oss.LDE, self._comment)

            if len(surfaces) == 1:
                self._surface = surfaces[0]

                return True

        return False

    # Mapping between OpticStudio surface properties and class properties
    _SURFACE_PROPERTY_NAMES: dict[str, str] = {
        "Comment": "_comment",
        "Radius": "_radius",
        "Thickness": "_thickness",
        "SemiDiameter": "_semi_diameter",
        "Conic": "_conic",
        "IsStop": "_is_stop",
    }

    def _get_surface_property(self, name: str):
        if self._is_built:
            return getattr(self.surface, name)

        return getattr(self, self._SURFACE_PROPERTY_NAMES[name])

    def _set_surface_property(self, name: str, value: Any):
        if self._is_built:
            setattr(self.surface, name, value)
        else:
            setattr(self, self._SURFACE_PROPERTY_NAMES[name], value)


@dataclass
class EyeGeometry:
    """Geometric parameters of an eye.

    Geometric parameters of an eye, defaulting to the Navarro model.
    Sizes are specified in mm.

    Attributes
    ----------
    axial_length : float
        Axial length of the eye, measured from cornea front to retina.
    lens_thickness : float
        Thickness of the crystalline lens
    lens_front_curvature : float
        Radius of curvature of the frontal lens surface
    lens_back_curvature : float
        Radius of curvature of the back lens surface
    iris_radius : float
        Radius of the iris
    anterior_chamber_depth : float
        Depth of the anterior chamber
    cornea_thickness : float
        Thickness of the cornea
    cornea_front_curvature : float
        Radius of curvature of the frontal cornea surface
    cornea_back_curvature : float
        Radius of curvature of the back cornea surface
    vitreous_thickness : float
        Thickness of the vitreous. This parameter is calculated from the other parameters.
    """

    axial_length: float = 23.9203
    lens_thickness: float = 4.0
    lens_back_curvature: float = -6.0
    lens_back_asphericity: float = -1
    lens_front_curvature: float = 10.2
    lens_front_asphericity: float = -3.1316
    iris_radius: float = 1.348
    anterior_chamber_depth: float = 3.05
    cornea_thickness: float = 0.55
    cornea_back_curvature: float = 6.50
    cornea_back_asphericity: float = 0
    cornea_front_curvature: float = 7.72
    cornea_front_asphericity: float = -0.26
    retina_curvature: float = -12
    retina_asphericity: float = 0

    estimate_cornea_back: InitVar[bool] = False

    vitreous_thickness: float = field(init=False)

    def __post_init__(self, estimate_cornea_back: bool):
        self.vitreous_thickness = self.axial_length - (
            self.cornea_thickness + self.anterior_chamber_depth + self.lens_thickness
        )

        if estimate_cornea_back:
            self.cornea_back_curvature = 0.81 * self.cornea_front_curvature


@dataclass
class EyeMaterials:
    """Material parameters of an eye.

    Material parameters of an eye, defaulting to the Navarro model.

    Attributes
    ----------
    cornea_refractive_index : float
        Refractive index of the cornea.
    aqueous_refractive_index : float
        Refractive index of the aqueous_refractive_index humour.
    lens_refractive_index : float
        Refractive index of the crystalline lens.
    vitreous_refractive_index : float
        Refractive index of the vitreous humour.
    """

    cornea_refractive_index: float = 1.3777
    aqueous_refractive_index: float = 1.3391
    lens_refractive_index: float = 1.4222
    vitreous_refractive_index: float = 1.3377


class BaseEye(ABC):
    @abstractmethod
    def __init__(self, parameters: EyeGeometry, materials: EyeMaterials):
        ...

    @abstractmethod
    def build(self, oss: OpticStudioSystem, start_from_index: int, replace_existing: bool):
        """Create the eye in OpticStudio.

        Create the eye model in the provided `OpticStudioSystem` `oss`, starting from `start_from_index`.
        The iris (pupil) is located at the STOP surface, and the retina at the IMAGE surface. For the other
        parts, new surfaces will be inserted by default. If `replace_existing` is set to `True`, existing
        surfaces will be overwritten.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the eye model is created.
        start_from_index : int
            Index at which the first  surface of the eye is located.
        replace_existing : bool
            If `True`, replaces existing surfaces instead of inserting new ones. Defaults to `False`.

        Raises
        ------
        AssertionError
            If the retina is not located at the IMAGE surface.
        """
        ...

    @property
    def surfaces(self) -> dict[str, Surface]:
        """Dictionary with surface names as keys and surfaces as values."""
        return {k: v for k, v in self.__dict__.items() if isinstance(v, Surface)}

    def update_surfaces(self, attribute: str, value: Any, surfaces: list[str] = None) -> None:
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
        surfaces = [self.surfaces[s] for s in surfaces] if surfaces is not None else self.surfaces.keys()

        for s in surfaces:
            setattr(s, attribute, value)

    def relink_surfaces(self, oss: OpticStudioSystem) -> bool:
        """Link surfaces to OpticStudio surfaces based on their comments.

        Attempt to re-link the surfaces of the eye to surfaces defined in OpticStudio, using the surface's comments.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem

        Returns
        -------
        bool
            `True` if all surfaces could be relinked, `False` otherwise.
        """
        result = [s.relink_surface(oss) for s in self.surfaces.values()]

        return all(result)


class Eye(BaseEye):
    def __init__(self, geometry: EyeGeometry = EyeGeometry(), materials: EyeMaterials = EyeMaterials()):
        self._cornea_front = Surface(
            comment="cornea front",
            radius=geometry.cornea_front_curvature,
            conic=geometry.cornea_front_asphericity,
            thickness=geometry.cornea_thickness,
            refractive_index=materials.cornea_refractive_index,
        )
        self._cornea_back = Surface(
            comment="cornea back / aqueous",
            radius=geometry.cornea_back_curvature,
            conic=geometry.cornea_back_asphericity,
            thickness=geometry.anterior_chamber_depth,
            refractive_index=materials.aqueous_refractive_index,
        )
        self._iris = Surface(
            comment="iris",
            is_stop=True,
            refractive_index=materials.aqueous_refractive_index,
            semi_diameter=geometry.iris_radius,
        )
        self._lens_front = Surface(
            comment="lens front",
            radius=geometry.lens_front_curvature,
            conic=-geometry.lens_front_asphericity,
            thickness=geometry.lens_thickness,
            refractive_index=materials.lens_refractive_index,
        )
        self._lens_back = Surface(
            comment="lens back / vitreous",
            radius=geometry.lens_back_curvature,
            conic=geometry.lens_back_asphericity,
            thickness=geometry.vitreous_thickness,
            refractive_index=materials.vitreous_refractive_index,
        )
        self._retina = Surface(
            comment="retina",
            radius=geometry.retina_curvature,
            conic=geometry.retina_asphericity,
            refractive_index=materials.vitreous_refractive_index,
        )

    @property
    def cornea_front(self) -> Surface:
        """Cornea front surface."""
        return self._cornea_front

    @cornea_front.setter
    def cornea_front(self, value: Surface) -> None:
        self._cornea_front = value

    @property
    def cornea_back(self) -> Surface:
        """Cornea back surface."""
        return self._cornea_back

    @cornea_back.setter
    def cornea_back(self, value: Surface) -> None:
        self._cornea_back = value

    @property
    def iris(self) -> Surface:
        """Iris / pupil surface."""
        return self._iris

    @iris.setter
    def iris(self, value: Surface) -> None:
        self._iris = value

    @property
    def lens_front(self) -> Surface:
        """Lens front surface."""
        return self._lens_front

    @lens_front.setter
    def lens_front(self, value: Surface) -> None:
        self._lens_front = value

    @property
    def lens_back(self) -> Surface:
        """Lens back surface."""
        return self._lens_back

    @lens_back.setter
    def lens_back(self, value: Surface) -> None:
        self._lens_back = value

    @property
    def retina(self) -> Surface:
        """Retina surface."""
        return self._retina

    @retina.setter
    def retina(self, value: Surface) -> None:
        self._retina = value

    def build(
        self,
        oss: OpticStudioSystem,
        start_from_index: int = 0,
        replace_existing: bool = False,
    ):
        """Create the eye in OpticStudio.

        Create the eye model in the provided `OpticStudioSystem` `oss`, starting from `start_from_index`.
        The iris (pupil) is located at the STOP surface, and the retina at the IMAGE surface. For the other
        parts, new surfaces will be inserted by default. If `replace_existing` is set to `True`, existing
        surfaces will be overwritten.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the eye model is created.
        start_from_index : int
            Index at which the first  surface of the eye is located.
        replace_existing : bool
            If `True`, replaces existing surfaces instead of inserting new ones. Defaults to `False`.

        Raises
        ------
        AssertionError
            If the retina is not located at the IMAGE surface.
        """
        self.cornea_front.build(oss, start_from_index + 1, replace_existing)
        self.cornea_back.build(oss, start_from_index + 2, replace_existing)
        self.iris.build(oss, start_from_index + 3, replace_existing=True)
        self.lens_front.build(oss, start_from_index + 4, replace_existing)
        self.lens_back.build(oss, start_from_index + 5, replace_existing)
        self.retina.build(oss, start_from_index + 6, replace_existing=True)

        # Sanity checks
        assert self.retina.surface.IsImage, "The retina is not located at the image position"
