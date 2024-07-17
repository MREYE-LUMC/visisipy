from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Generic, TypeVar, Union
from warnings import warn

import zospy as zp

from visisipy.models import BaseSurface
from visisipy.models.geometry import StandardSurface, Stop, Surface
from visisipy.models.materials import MaterialModel

if TYPE_CHECKING:
    from zospy.api import _ZOSAPI
    from zospy.zpcore import OpticStudioSystem

__all__ = ("OpticStudioSurface", "make_surface")

PropertyType = TypeVar("PropertyType")


class OpticStudioSurfaceProperty(Generic[PropertyType]):
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: OpticStudioSurface, objtype=None) -> PropertyType:
        if obj.surface is None:
            return None

        return getattr(obj.surface, self.name)

    def __set__(self, obj: OpticStudioSurface, value: PropertyType) -> None:
        if obj.surface is None:
            message = f"Cannot set attribute {self.name} of non-built surface."
            raise AttributeError(message)

        setattr(obj.surface, self.name, value)


class OpticStudioSurface(BaseSurface):
    """
    Sequential surface in OpticStudio.
    """

    _TYPE: str = "Standard"

    def __init__(
        self,
        *,
        comment: str,
        radius: float = float("inf"),
        thickness: float = 0.0,
        semi_diameter: float | None = None,
        conic: float = 0.0,
        material: MaterialModel | str | None = None,
        is_stop: bool | None = None,
    ):
        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._material = material
        self._is_stop = is_stop

        self._surface = None
        self._is_built = False

    comment: str = OpticStudioSurfaceProperty("Comment")
    radius: float = OpticStudioSurfaceProperty("Radius")
    thickness: float = OpticStudioSurfaceProperty("Thickness")
    semi_diameter: float = OpticStudioSurfaceProperty("SemiDiameter")
    conic: float = OpticStudioSurfaceProperty("Conic")
    is_stop: bool = OpticStudioSurfaceProperty("IsStop")

    def _get_material(self) -> MaterialModel | str | None:
        if not self._is_built:
            return None

        if self.surface.MaterialCell.GetSolveData().Type == zp.constants.Editors.SolveType.MaterialModel:
            material_model = self.surface.MaterialCell.GetSolveData()._S_MaterialModel  # noqa: SLF001

            return MaterialModel(
                refractive_index=material_model.IndexNd,
                abbe_number=material_model.AbbeVd,
                partial_dispersion=material_model.dPgF,
            )

        return self.surface.Material

    def _set_material(self, material: MaterialModel | str | None) -> None:
        if material is None:  # Do nothing if material is None
            return

        if isinstance(material, MaterialModel):
            zp.solvers.material_model(
                self.surface.MaterialCell,
                refractive_index=material.refractive_index,
                abbe_number=material.abbe_number,
                partial_dispersion=material.partial_dispersion,
            )
        elif isinstance(material, str):
            self.surface.Material = material
        else:
            raise TypeError("'material' must be MaterialModel or str.")

    @property
    def material(self) -> MaterialModel | str:
        return self._get_material()

    @material.setter
    def material(self, material: MaterialModel | str) -> None:
        if self._is_built:
            self._set_material(material)

    @property
    def surface(self) -> _ZOSAPI.Editors.LDE.ILDERow | None:
        """OpticStudio surface object.

        This property only has a value if the surface has been built with `Surface.build`.
        """
        return self._surface

    def _set_surface_type(self):
        surface_type = zp.constants.process_constant(zp.constants.Editors.LDE.SurfaceType, self._TYPE)

        if self.surface is not None and self.surface.Type != surface_type:
            zp.functions.lde.surface_change_type(self.surface, self._TYPE)

    def build(self, oss: OpticStudioSystem, *, position: int, replace_existing: bool = False):
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` `oss` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.
        """
        self._surface = oss.LDE.GetSurfaceAt(position) if replace_existing else oss.LDE.InsertNewSurfaceAt(position)

        self._set_surface_type()

        self.comment = self._comment
        self.radius = self._radius
        self.thickness = self._thickness
        self.conic = self._conic

        self._set_material(self._material)

        # Only set semi_diameter when explicitly specified
        if self._semi_diameter is not None:
            self.semi_diameter = self._semi_diameter

        # Only set IsStop when explicitly specified
        if self._is_stop is True:
            self.is_stop = self._is_stop
        elif self._is_stop is False:
            warn(
                "is_stop is set to False, but this is not supported in OpticStudio. Explicitly setting is_stop will "
                "always convert the surface to a stop. This setting has been ignored."
            )

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


@singledispatch
def make_surface(surface: Surface, material: str | MaterialModel, comment: str = "") -> OpticStudioSurface:
    """Create an `OpticStudioSurface` instance from a given `Surface` instance.

    Parameters
    ----------
    surface : Surface
        The Surface instance from which to create the OpticStudioSurface instance.
    material : str | MaterialModel
        The material of the surface. This can be either a string representing the name
        of the material or a MaterialModel instance.
    comment : str, optional
        A comment to be associated with the surface. This is an empty string by default.

    Returns
    -------
    OpticStudioSurface
        The created OpticStudioSurface instance.
    """
    return OpticStudioSurface(comment=comment, thickness=surface.thickness, material=material)


@make_surface.register
def _make_surface(
    surface: StandardSurface,
    material: Union[str, MaterialModel],  # noqa: UP007
    comment: str = "",
) -> OpticStudioSurface:
    return OpticStudioSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        material=material,
    )


@make_surface.register
def _make_surface(surface: Stop, material: Union[str, MaterialModel] = "", comment: str = "") -> OpticStudioSurface:  # noqa: UP007
    return OpticStudioSurface(
        comment=comment,
        thickness=surface.thickness,
        material=material,
        semi_diameter=surface.semi_diameter,
        is_stop=True,
    )
