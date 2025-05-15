"""Optical surfaces for Optiland."""

from __future__ import annotations

import weakref
from functools import reduce, singledispatch
from operator import attrgetter
from typing import TYPE_CHECKING, Generic, TypeVar, Union

import optiland.materials
from optiland.materials import AbbeMaterial, IdealMaterial, Material

from visisipy.models import BaseSurface
from visisipy.models import NoSurface as OptilandNoSurface
from visisipy.models.geometry import (
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel

if TYPE_CHECKING:
    import optiland.surfaces
    from optiland.optic import Optic


PropertyType = TypeVar("PropertyType")


class OptilandSurfaceProperty(Generic[PropertyType]):
    """Descriptor for Optiland surface properties.

    This descriptor is used to access and set properties of the Optiland surface.
    """

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
                reduce(getattr, name_split[:-1]),
                name_split[-1],
                value,
            )


class _built_only_property(property):  # noqa: N801
    """Property that can only be accessed after the surface has been built."""

    def __get__(self, obj: OptilandSurface, objtype=None):
        if not obj._is_built:  # noqa: SLF001
            return None

        return super().__get__(obj, objtype)

    def __set__(self, obj: OptilandSurface, value) -> None:
        if not obj._is_built:  # noqa: SLF001
            message = "Cannot set attribute of non-built surface."
            raise AttributeError(message)

        super().__set__(obj, value)


class OptilandSurface(BaseSurface):
    """Sequential surface in Optiland.

    Note that, unlike OpticStudio surfaces, it is not possible to update surfaces after they have been built.
    """

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
        """Create a new Optiland surface.

        Parameters
        ----------
        comment : str
            Comment for the surface.
        radius : float
            Radius of curvature of the surface, in mm. Defaults to infinity (flat surface).
        thickness : float
            Thickness of the surface, in mm. Defaults to 0.0 mm.
        semi_diameter : float, optional
            Semi-diameter of the surface aperture, in mm. Defaults to `None`, which means the semi-diameter will be
            determined by Optiland.
        conic : float
            Conic constant of the surface. Defaults to 0.0 (spherical surface).
        material : MaterialModel | str, optional
            Material of the surface. This can be either a string representing the name of the material or a
            MaterialModel instance. Defaults to `None`, which means the material is assumed to be air.
        is_stop : bool, optional
            If `True`, the surface is treated as a stop.
        """
        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._material = material
        self._is_stop = is_stop

        self._surface: optiland.surfaces.Surface | None = None
        self._optic: Optic | None = None
        self._index: int | None = None
        self._is_built: bool = False

    @property
    def surface(self) -> optiland.surfaces.Surface | None:
        """Optiland surface object.

        This property only has a value if the surface has been built.
        """
        return self._surface

    @_built_only_property
    def comment(self) -> str:
        """Comment for the surface."""
        return self.surface.comment  # type: ignore

    @comment.setter
    def comment(self, value: str) -> None:
        """Set the comment for the surface."""
        self.surface.comment = value  # type: ignore

    @_built_only_property
    def radius(self) -> float:
        """Radius of the surface."""
        return self._optic.surface_group.radii[self._index]  # type: ignore

    @_built_only_property
    def thickness(self) -> float:
        """Thickness of the surface."""
        if self._index == self._optic.surface_group.num_surfaces - 1:
            # Last surface in the system, return 0.0
            return 0.0

        return self._optic.surface_group.get_thickness(self._index)[0]  # type: ignore

    @_built_only_property
    def semi_diameter(self) -> float | None:
        """Semi-diameter of the surface."""
        return self.surface.semi_aperture  # type: ignore

    @_built_only_property
    def conic(self) -> float:
        """Conic constant of the surface."""
        return self._optic.surface_group.conic[self._index]  # type: ignore

    @_built_only_property
    def material(self) -> MaterialModel | str | None:
        """Material of the surface."""
        return self._get_material()

    @_built_only_property
    def is_stop(self) -> bool:
        """Flag indicating if the surface is a stop."""
        return self.surface.is_stop  # type: ignore

    def _get_material(self) -> MaterialModel | str | None:
        """Get the material of the surface."""
        if not self._is_built:
            return None

        if isinstance(self.surface.material_post, IdealMaterial):  # type: ignore
            return MaterialModel(
                refractive_index=self.surface.material_post.index,  # type: ignore
                abbe_number=0.0,
                partial_dispersion=0.0,
            )

        if isinstance(self.surface.material_post, AbbeMaterial):  # type: ignore
            return MaterialModel(
                refractive_index=self.surface.material_post.index,  # type: ignore
                abbe_number=self.surface.material_post.abbe,  # type: ignore
                partial_dispersion=0.0,
            )

        if isinstance(self.surface.material_post, Material):  # type: ignore
            return self.surface.material_post.name  # type: ignore

        raise TypeError(
            f"Unsupported material type: {type(self.surface.material_post)}"  # type: ignore
        )

    @staticmethod
    def _convert_material(
        material: MaterialModel | str | None,
    ) -> str | IdealMaterial | AbbeMaterial:
        if isinstance(material, str):
            return material

        if material is None:
            return "Air"

        if isinstance(material, MaterialModel):
            if material.abbe_number == 0:
                return IdealMaterial(n=material.refractive_index, k=0.0)

            return AbbeMaterial(n=material.refractive_index, abbe=material.abbe_number)

        raise TypeError("'material' must be MaterialModel or str.")

    def build(self, optic: Optic, *, position: int, replace_existing: bool = False) -> int:
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

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be after this index.
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

        if self._semi_diameter is not None:
            self.surface.set_semi_aperture(self._semi_diameter)  # type: ignore

        self._optic = weakref.proxy(optic)
        self._index = position
        self._is_built = True

        return position


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
    surface: ZernikeStandardSagSurface,
    material: Union[str, MaterialModel] = "",  # noqa: UP007
    comment: str = "",
) -> OptilandSurface:
    raise NotImplementedError("ZernikeStandardSagSurface is not supported in Optiland.")


@make_surface.register
def _make_surface(
    surface: ZernikeStandardPhaseSurface,
    material: Union[str, MaterialModel] = "",  # noqa: UP007
    comment: str = "",
) -> OptilandSurface:
    raise NotImplementedError("ZernikeStandardPhaseSurface is not supported in Optiland.")


@make_surface.register
def _make_surface(
    surface: NoSurface,  # noqa: ARG001
    material: None = None,  # noqa: ARG001
    comment: str = "",  # noqa: ARG001
) -> OptilandNoSurface:
    return OptilandNoSurface()
