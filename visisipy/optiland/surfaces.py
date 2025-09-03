"""Optical surfaces for Optiland."""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from functools import reduce, singledispatch
from operator import attrgetter
from typing import TYPE_CHECKING, Generic, TypeVar, Union, overload
from warnings import warn

from optiland.materials import AbbeMaterial, IdealMaterial, Material

from visisipy.models import BaseSurface
from visisipy.models import NoSurface as OptilandNoSurface
from visisipy.models.geometry import (
    BiconicSurface,
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel
from visisipy.types import TypedDict
from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    import optiland.surfaces
    from optiland.optic import Optic

    from visisipy.types import NotRequired, Unpack


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


class OptilandCommonSurfaceParameters(TypedDict):
    """Optiland common surface parameters."""

    comment: str
    thickness: NotRequired[float]
    material: NotRequired[IdealMaterial | AbbeMaterial | str]
    semi_diameter: float | None
    is_stop: bool


class OptilandStandardSurfaceParameters(OptilandCommonSurfaceParameters):
    """Optiland standard surface parameters."""

    radius: NotRequired[float]
    conic: NotRequired[float]


class OptilandBiconicSurfaceParameters(OptilandCommonSurfaceParameters):
    """Optiland biconic surface parameters."""

    radius_y: NotRequired[float]
    radius_x: NotRequired[float]
    conic_y: NotRequired[float]
    conic_x: NotRequired[float]


class OptilandZernikeSurfaceParameters(OptilandStandardSurfaceParameters):
    """Optiland Zernike surface parameters."""

    zernike_type: NotRequired[str]
    norm_radius: NotRequired[float]
    coefficients: NotRequired[list[float]]


class BaseOptilandSurface(BaseSurface, ABC):
    def __init__(self):
        """Initialize the base Optiland surface."""
        self._surface: optiland.surfaces.Surface | None = None
        self._optic: Optic | None = None
        self._index: int | None = None
        self._is_built: bool = False

    @property
    @abstractmethod
    def _TYPE(self) -> str: ...  # noqa: N802

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
    def material(self) -> MaterialModel | str | None:
        """Material of the surface."""
        return self._get_material()

    @_built_only_property
    def is_stop(self) -> bool:
        """Flag indicating if the surface is a stop."""
        return self.surface.is_stop  # type: ignore

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

    @overload
    def _create_surface(
        self,
        optic: Optic,
        *,
        position: int,
        replace_existing: bool = False,
        **kwargs: Unpack[OptilandStandardSurfaceParameters],
    ) -> optiland.surfaces.Surface: ...

    @overload
    def _create_surface(
        self,
        optic: Optic,
        *,
        position: int,
        replace_existing: bool = False,
        **kwargs: Unpack[OptilandBiconicSurfaceParameters],
    ) -> optiland.surfaces.Surface: ...

    @overload
    def _create_surface(
        self,
        optic: Optic,
        *,
        position: int,
        replace_existing: bool = False,
        **kwargs: Unpack[OptilandZernikeSurfaceParameters],
    ) -> optiland.surfaces.Surface: ...

    def _create_surface(
        self, optic: Optic, *, position: int, replace_existing: bool = False, **kwargs
    ) -> optiland.surfaces.Surface:
        """Helper method to create a surface in Optiland.

        This method adds a surface to the provided `Optic` object at the specified `position`, and handles
        all common surface creation logic. Surface parameters are passed as keyword arguments.

        Parameters
        ----------
        optic : Optic
            The Optic object to which the surface will be added.
        position : int
            The index at which the surface will be added, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.
        **kwargs : Unpack[OptilandStandardSurfaceParameters] | Unpack[OptilandBiconicSurfaceParameters]
            Keyword arguments for the surface parameters.

        Returns
        -------
        optiland.surfaces.Surface
            The created surface object.

        See Also
        --------
        OptilandStandardSurfaceParameters : Parameters for standard surfaces in Optiland.
        OptilandBiconicSurfaceParameters : Parameters for biconic surfaces in Optiland.
        """
        optic.add_surface(index=position, surface_type=self._TYPE, **kwargs)

        if replace_existing:
            optic.surface_group.surfaces.pop(position + 1)

        self._surface = optic.surface_group.surfaces[position]

        if kwargs["semi_diameter"] is not None:
            self.surface.set_semi_aperture(kwargs["semi_diameter"])  # type: ignore

        self._optic = weakref.proxy(optic)
        self._index = position
        self._is_built = True

        return self._surface


class OptilandSurface(BaseOptilandSurface):
    """Sequential surface in Optiland.

    Note that, unlike OpticStudio surfaces, it is not possible to update surfaces after they have been built.
    """

    _TYPE: str = "standard"

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
        super().__init__()

        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._material = material
        self._is_stop = is_stop

    @_built_only_property
    def radius(self) -> float:
        """Radius of the surface."""
        return self._optic.surface_group.radii[self._index]  # type: ignore

    @_built_only_property
    def conic(self) -> float:
        """Conic constant of the surface."""
        return self._optic.surface_group.conic[self._index]  # type: ignore

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
        self._create_surface(
            optic=optic,
            position=position,
            replace_existing=replace_existing,
            comment=self._comment,
            radius=self._radius,
            thickness=self._thickness,
            semi_diameter=self._semi_diameter,
            conic=self._conic,
            material=self._convert_material(self._material),
            is_stop=bool(self._is_stop),
        )

        return position


class OptilandBiconicSurface(BaseOptilandSurface):
    """Biconic surface in Optiland."""

    _TYPE: str = "biconic"

    def __init__(
        self,
        comment: str,
        *,
        radius: float = float("inf"),
        radius_x: float = float("inf"),
        thickness: float = 0.0,
        semi_diameter: float | None = None,
        conic: float = 0.0,
        conic_x: float = 0.0,
        material: MaterialModel | str | None = None,
        is_stop: bool | None = None,
    ):
        """Create a new biconic Optiland surface.

        Parameters
        ----------
        comment : str
            Comment for the surface.
        radius : float
            Radius of curvature of the surface in the Y direction, in mm. Defaults to infinity (flat surface).
        radius_x : float
            Radius of curvature of the surface in the X direction, in mm. Defaults to infinity (flat surface).
        thickness : float
            Thickness of the surface, in mm. Defaults to 0.0 mm.
        semi_diameter : float, optional
            Semi-diameter of the surface aperture, in mm. Defaults to `None`, which means the semi-diameter will be
            determined by Optiland.
        conic : float
            Conic constant of the surface in the Y direction. Defaults to 0.0 (spherical surface).
        conic_x : float
            Conic constant of the surface in the X direction. Defaults to 0.0 (spherical surface).
        material : MaterialModel | str, optional
            Material of the surface. This can be either a string representing the name of the material or a
            MaterialModel instance. Defaults to `None`, which means the material is assumed to be air.
        is_stop : bool, optional
            If `True`, the surface is treated as a stop.
        """
        super().__init__()

        self._comment = comment
        self._radius = radius
        self._radius_x = radius_x
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._conic_x = conic_x
        self._material = material
        self._is_stop = is_stop

    @_built_only_property
    def radius(self) -> float:
        """Radius of the surface in the Y direction."""
        return float(self.surface.geometry.Ry)  # type: ignore

    @_built_only_property
    def conic(self) -> float:
        """Conic constant of the surface in the X direction."""
        return float(self.surface.geometry.ky)  # type: ignore

    @_built_only_property
    def radius_x(self) -> float:
        """Radius of the surface in the X direction."""
        return float(self.surface.geometry.Rx)  # type: ignore

    @_built_only_property
    def conic_x(self) -> float:
        """Conic constant of the surface in the X direction."""
        return float(self.surface.geometry.kx)  # type: ignore

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
        self._create_surface(
            optic=optic,
            position=position,
            replace_existing=replace_existing,
            comment=self._comment,
            radius_y=self._radius,
            radius_x=self._radius_x,
            thickness=self._thickness,
            semi_diameter=self._semi_diameter,
            conic_y=self._conic,
            conic_x=self._conic_x,
            material=self._convert_material(self._material),
            is_stop=bool(self._is_stop),
        )

        return position


class OptilandZernikeStandardSagSurface(BaseOptilandSurface):
    """Zernike Standard Sag surface in Optiland."""

    _TYPE: str = "zernike"

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
        number_of_terms: int = 0,
        norm_radius: float = 100,
        zernike_coefficients: ZernikeCoefficients | dict[int, float] | None = None,
    ) -> None:
        """Create a new Zernike Optiland surface."""
        super().__init__()

        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._material = material
        self._is_stop = is_stop
        self._norm_radius = norm_radius

        if zernike_coefficients is not None:
            if any(key > number_of_terms for key in zernike_coefficients):
                raise ValueError(
                    f"Zernike coefficients must be less than or equal to the maximum term {number_of_terms}."
                )
            if any(key < 1 for key in zernike_coefficients):
                raise ValueError("Zernike coefficients must be positive integers.")

        self._number_of_terms = number_of_terms
        self._zernike_coefficients = (
            zernike_coefficients
            if zernike_coefficients is not None
            else ZernikeCoefficients({i: 0 for i in range(1, number_of_terms + 1)})
        )

    @_built_only_property
    def radius(self) -> float:
        """Radius of the surface."""
        return self.surface.geometry.radius  # type: ignore

    @_built_only_property
    def conic(self) -> float:
        """Conic constant of the surface."""
        return self.surface.geometry.k  # type: ignore

    @_built_only_property
    def norm_radius(self) -> float:
        """Normalization radius of the Zernike surface."""
        return self.surface.geometry.norm_radius  # type: ignore

    @_built_only_property
    def coefficients(self) -> ZernikeCoefficients:
        """Zernike coefficients of the surface."""
        coefficients = self.surface.geometry.coefficients  # type: ignore
        return ZernikeCoefficients(dict(enumerate(coefficients, start=1)))

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
        coefficients = [self._zernike_coefficients.get(i, 0.0) for i in range(1, self._number_of_terms + 1)]

        self._create_surface(
            optic=optic,
            position=position,
            replace_existing=replace_existing,
            comment=self._comment,
            radius=self._radius,
            thickness=self._thickness,
            semi_diameter=self._semi_diameter,
            conic=self._conic,
            material=self._convert_material(self._material),
            is_stop=bool(self._is_stop),
            norm_radius=self._norm_radius,
            zernike_type="noll",
            coefficients=coefficients,
        )

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
    surface: BiconicSurface,
    material: Union[str, MaterialModel] = "Air",  # noqa: UP007
    comment: str = "",
) -> OptilandBiconicSurface:
    return OptilandBiconicSurface(
        comment=comment,
        radius=surface.radius,
        radius_x=surface.radius_x,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        conic_x=surface.asphericity_x,
        material=material,
    )


@make_surface.register
def _make_surface(
    surface: ZernikeStandardSagSurface,
    material: Union[str, MaterialModel] = "",  # noqa: UP007
    comment: str = "",
) -> OptilandZernikeStandardSagSurface:
    if surface.extrapolate:
        warn("Zernike surface extrapolation is not supported in Optiland.", UserWarning)
    if surface.zernike_decenter_x != 0.0 or surface.zernike_decenter_y != 0.0:
        warn("Zernike surface decentering is not supported in Optiland.", UserWarning)

    return OptilandZernikeStandardSagSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        material=material,
        number_of_terms=surface.maximum_term,
        norm_radius=surface.norm_radius,
        zernike_coefficients=surface.zernike_coefficients,
        is_stop=surface.is_stop,
    )


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
