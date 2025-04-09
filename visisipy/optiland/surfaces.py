from __future__ import annotations

from functools import reduce, singledispatch
from operator import attrgetter
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import optiland.materials
from optiland.materials import AbbeMaterial, IdealMaterial

from visisipy.models import BaseSurface
from visisipy.models.geometry import (
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel  # noqa: TCH001

if TYPE_CHECKING:
    import optiland.surfaces
    from optiland.optic import Optic


PropertyType = TypeVar("PropertyType")


class OptilandSurfaceProperty(Generic[PropertyType]):
    def __init__(self, name: str):
        self.name = name
        self.attrgetter = attrgetter(name)

    def __get__(self, obj: OptilandSurface, objtype=None) -> PropertyType:
        if obj.surface is None:
            return None

        return self.attrgetter(obj.surface)

    def __set__(self, obj: OptilandSurface, value: PropertyType) -> None:
        if obj.surface is None:
            return

        name_split = self.name.split(".")

        if len(name_split) == 1:
            setattr(obj.surface, self.name, value)
        else:
            setattr(
                reduce(lambda o, a: getattr(o, a), name_split[:-1]),
                name_split[-1],
                value,
            )


class OptilandSurface(BaseSurface):
    """Sequential surface in Optiland."""

    _TYPE: str = "Standard"

    def __init__(
        self,
        comment: str,
        *,
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

    @property
    def surface(self) -> optiland.surfaces.Surface | None:
        """Optiland surface object.

        This property only has a value if the surface has been built.
        """
        return self._surface

    @staticmethod
    def _convert_material(
        material: MaterialModel | str | None,
    ) -> str | IdealMaterial | AbbeMaterial:
        if isinstance(material, str):
            return material

        if material is None:
            return "Air"

        if material.abbe_number == 0:
            return IdealMaterial(n=material.refractive_index, k=0.0)

        return AbbeMaterial(n=material.refractive_index, abbe=material.abbe_number)

    def build(self, optic: Optic, *, position: int, replace_existing: bool = False):
        """Create the surface in Optiland.

        Create the surface in the provided `Optic` object at the specified `position`.
        By default, a new surface will be created. If `replace_existing` is `True`, the existing surface at the
        specified position will be replaced.

        Parameters
        ----------
        optic : Optic
            The Optic object to which the surface will be added.
        position : int
            The index at which the surface will be added, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.
        """
        optic.add_surface(
            index=position,
            radius=self._radius,
            thickness=self._thickness,
            conic=self._conic,
            material=self._convert_material(self._material),
            is_stop=bool(self._is_stop),
            comment=self._comment,
        )

        if replace_existing:
            optic.surface_group.surfaces.pop(position + 1)

        self._surface = optic.surface_group.surfaces[position]

        self._is_built = True


@singledispatch
def make_surface(surface: Surface, material: str | MaterialModel, comment: str = "") -> OptilandSurface:
    """Create an `OptilandSurface` instance from a given `Surface` instance.

    Parameters
    ----------
    surface : Surface
        The Surface instance from which to create the OptilandSurface instance.
    material : str | MaterialModel
        The material of the surface. This can be either a string representing the name
        of the material or a MaterialModel instance.
    comment : str, optional
        A comment to be associated with the surface. This is an empty string by default.

    Returns
    -------
    OptilandSurface
        The created OptilandSurface instance.
    """
    return OptilandSurface(comment=comment, thickness=surface.thickness, material=material)


@make_surface.register
def _make_surface(
    surface: StandardSurface,
    material: Union[str, MaterialModel],  # noqa: UP007
    comment: str = "",
) -> OptilandSurface:
    return OptilandSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        material=material,
    )


@make_surface.register
def _make_surface(
    surface: Stop,
    material: Union[str, MaterialModel] = "Air",  # noqa: UP007
    comment: str = "",
) -> OptilandSurface:
    return OptilandSurface(
        comment=comment,
        thickness=surface.thickness,
        material=material,
        semi_diameter=surface.semi_diameter,
        is_stop=True,
    )


@make_surface.register
def _make_surface(
    surface: ZernikeStandardSagSurface,  # noqa: ARG001
    material: Union[str, MaterialModel] = "",  # noqa: UP007, ARG001
    comment: str = "",  # noqa: ARG001
) -> OptilandSurface:
    raise NotImplementedError("ZernikeStandardSagSurface is not supported in Optiland.")


@make_surface.register
def _make_surface(
    surface: ZernikeStandardPhaseSurface,  # noqa: ARG001
    material: Union[str, MaterialModel] = "",  # noqa: UP007, ARG001
    comment: str = "",  # noqa: ARG001
) -> OptilandSurface:
    raise NotImplementedError("ZernikeStandardPhaseSurface is not supported in Optiland.")
