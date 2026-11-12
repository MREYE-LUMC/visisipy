"""Optical surfaces for OpticStudio."""

from __future__ import annotations

from abc import ABC
from functools import singledispatch
from typing import TYPE_CHECKING, Generic, TypeVar
from warnings import warn

import zospy as zp

from visisipy.models.base import BaseSurface
from visisipy.models.base import NoSurface as OpticStudioNoSurface
from visisipy.models.geometry import (
    BiconicSurface,
    EvenAsphereSurface,
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeCoefficients,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from zospy.api import _ZOSAPI
    from zospy.zpcore import OpticStudioSystem

__all__ = ("OpticStudioSurface", "OpticStudioZernikeStandardSagSurface", "make_surface")

PropertyType = TypeVar("PropertyType")


class OpticStudioSurfaceProperty(Generic[PropertyType]):
    """Descriptor for OpticStudio surface properties.

    This descriptor is used to access and modify properties of the OpticStudio surface.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: OpticStudioSurface, objtype=None) -> PropertyType | None:
        if obj.surface is None:
            return None

        return getattr(obj.surface, self.name)

    def __set__(self, obj: OpticStudioSurface, value: PropertyType) -> None:
        if obj.surface is None:
            message = f"Cannot set attribute {self.name} of non-built surface."
            raise AttributeError(message)

        setattr(obj.surface, self.name, value)


class OpticStudioSurfaceDataProperty(Generic[PropertyType]):
    """Property descriptor for OpticStudio surface data properties.

    This descriptor is used to access and modify properties of the OpticStudio surface data.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, obj: OpticStudioSurface, objtype=None) -> PropertyType | None:
        if obj.surface is None:
            return None

        return getattr(obj.surface.SurfaceData, self.name)

    def __set__(self, obj: OpticStudioSurface, value: PropertyType) -> None:
        if obj.surface is None:
            message = f"Cannot set attribute {self.name} of non-built surface."
            raise AttributeError(message)

        setattr(obj.surface.SurfaceData, self.name, value)


class OpticStudioSurface(BaseSurface):
    """Sequential surface in OpticStudio."""

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
        """Create a new OpticStudio surface.

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
            determined by OpticStudio.
        conic : float
            Conic constant of the surface. Defaults to 0.0 (spherical surface).
        material : MaterialModel | str, optional
            Material of the surface. This can be either a string representing the name of the material or a
            MaterialModel instance. Defaults to `None`, which means no material is set.
        is_stop : bool, optional
            If `True`, the surface is treated as a stop. Defaults to `None`, which means the stop status will be
            determined by OpticStudio. Note that setting the `IsStop` property in OpticStudio will always convert the
            surface to a stop, regardless of the value. When set to `False`, this parameter will be ignored.
        """
        self._comment = comment
        self._radius = radius
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._material = material
        self._is_stop = is_stop

        self._surface = None
        self._is_built = False

    comment: OpticStudioSurfaceProperty[str] = OpticStudioSurfaceProperty("Comment")
    radius: OpticStudioSurfaceProperty[float] = OpticStudioSurfaceProperty("Radius")
    thickness: OpticStudioSurfaceProperty[float] = OpticStudioSurfaceProperty("Thickness")
    semi_diameter: OpticStudioSurfaceProperty[float] = OpticStudioSurfaceProperty("SemiDiameter")
    conic: OpticStudioSurfaceProperty[float] = OpticStudioSurfaceProperty("Conic")
    is_stop: OpticStudioSurfaceProperty[bool] = OpticStudioSurfaceProperty("IsStop")

    def _get_material(self) -> MaterialModel | str | None:
        if not self._is_built:
            return None

        if self.surface.MaterialCell.GetSolveData().Type == zp.constants.Editors.SolveType.MaterialModel:
            material_model = self.surface.MaterialCell.GetSolveData()._S_MaterialModel  # ruff: ignore[private-member-access]

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
    def material(self) -> MaterialModel | str | None:
        """Material of the surface."""
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

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        self._surface = (
            optical_system.LDE.GetSurfaceAt(position)
            if replace_existing
            else optical_system.LDE.InsertNewSurfaceAt(position)
        )

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

        return self.surface.SurfaceNumber

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


class OpticStudioBiconicSurface(OpticStudioSurface):
    """Biconic surface in OpticStudio."""

    _TYPE: str = "Biconic"

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
        """Create a new biconic OpticStudio surface.

        Parameters
        ----------
        comment : str
            Comment for the surface.
        radius : float
            Radius of curvature in the Y direction of the surface, in mm. Defaults to infinity (flat surface).
        radius_x : float
            Radius of curvature in the X direction of the surface, in mm. Defaults to infinity (flat surface).
        thickness : float
            Thickness of the surface, in mm. Defaults to 0.0 mm.
        semi_diameter : float, optional
            Semi-diameter of the surface aperture, in mm. Defaults to `None`, which means the semi-diameter will be
            determined by OpticStudio.
        conic : float
            Conic constant of the surface in the Y direction. Defaults to 0.0 (spherical surface).
        conic_x : float
            Conic constant of the surface in the X direction. Defaults to 0.0 (spherical surface).
        material : MaterialModel | str, optional
            Material of the surface. This can be either a string representing the name of the material or a
            MaterialModel instance. Defaults to `None`, which means no material is set.
        is_stop : bool, optional
            If `True`, the surface is treated as a stop. Defaults to `None`, which means the stop status will be
            determined by OpticStudio. Note that setting the `IsStop` property in OpticStudio will always convert the
            surface to a stop, regardless of the value. When set to `False`, this parameter will be ignored.
        """
        self._comment = comment
        self._radius = radius
        self._radius_x = radius_x
        self._thickness = thickness
        self._semi_diameter = semi_diameter
        self._conic = conic
        self._conic_x = conic_x
        self._material = material
        self._is_stop = is_stop

        self._surface = None
        self._is_built = False

    radius_x: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("XRadius")
    conic_x: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("XConic")

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        surface_index = super().build(optical_system, position=position, replace_existing=replace_existing)

        self.radius_x = self._radius_x
        self.conic_x = self._conic_x

        return surface_index


class OpticStudioEvenAsphereSurface(OpticStudioSurface):
    """Even asphere surface in OpticStudio.

    OpticStudio supports at most 8 asphere coefficients, i.e. polynomial terms up to :math:`r^{16}`.
    After the surface has been built, individual coefficients can be retrieved and modified with
    `get_asphere_coefficient` and `set_asphere_coefficient`.
    """

    _TYPE: str = "EvenAspheric"
    MAX_ASPHERE_COEFFICIENTS: int = 8

    def __init__(
        self,
        comment: str,
        *,
        radius: float = float("inf"),
        thickness: float = 0.0,
        semi_diameter: float | None = None,
        conic: float = 0.0,
        asphere_coefficients: Sequence[float] | None = None,
        material: MaterialModel | str | None = None,
        is_stop: bool | None = None,
    ):
        """Create a new even asphere OpticStudio surface.

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
            determined by OpticStudio.
        conic : float
            Conic constant of the surface. Defaults to 0.0 (spherical surface).
        asphere_coefficients : Sequence[float], optional
            Coefficients of the even-order polynomial terms. The first element is the coefficient of :math:`r^2`,
            the second element the coefficient of :math:`r^4`, etc. At most 8 coefficients can be specified.
            Defaults to `None`, which means no polynomial terms are used.
        material : MaterialModel | str, optional
            Material of the surface. This can be either a string representing the name of the material or a
            MaterialModel instance. Defaults to `None`, which means no material is set.
        is_stop : bool, optional
            If `True`, the surface is treated as a stop. Defaults to `None`, which means the stop status will be
            determined by OpticStudio. Note that setting the `IsStop` property in OpticStudio will always convert the
            surface to a stop, regardless of the value. When set to `False`, this parameter will be ignored.

        Raises
        ------
        ValueError
            If more than 8 asphere coefficients are specified.
        """
        super().__init__(
            comment=comment,
            radius=radius,
            thickness=thickness,
            semi_diameter=semi_diameter,
            conic=conic,
            material=material,
            is_stop=is_stop,
        )

        coefficients = list(asphere_coefficients) if asphere_coefficients is not None else []

        if len(coefficients) > self.MAX_ASPHERE_COEFFICIENTS:
            raise ValueError(
                f"OpticStudio supports at most {self.MAX_ASPHERE_COEFFICIENTS} asphere coefficients, "
                f"got {len(coefficients)}."
            )

        self._asphere_coefficients = coefficients

    def _validate_coefficient(self, n: int) -> None:
        if not 1 <= n <= self.MAX_ASPHERE_COEFFICIENTS:
            raise ValueError(
                f"Asphere coefficient index must be between 1 and {self.MAX_ASPHERE_COEFFICIENTS}, got {n}."
            )

    @property
    def _surface_data(self) -> _ZOSAPI.Editors.LDE.ISurfaceEvenAspheric:
        surface = self.surface
        if surface is None:
            raise RuntimeError("The surface has not been built.")

        return surface.SurfaceData

    def get_asphere_coefficient(self, n: int) -> float:
        """Get the value of the nth asphere coefficient.

        Parameters
        ----------
        n : int
            Index of the asphere coefficient, starting at 1 for the coefficient of :math:`r^2`. Index 2 is the
            coefficient of :math:`r^4`, etc., up to index 8 for :math:`r^{16}`.

        Returns
        -------
        float
            The value of the asphere coefficient.

        Raises
        ------
        ValueError
            If `n` is smaller than 1 or larger than 8.
        RuntimeError
            If the surface has not been built.
        """
        self._validate_coefficient(n)

        return self._surface_data.GetNthEvenOrderTerm(n)

    def set_asphere_coefficient(self, n: int, value: float) -> None:
        """Set the value of the nth asphere coefficient.

        Other asphere coefficients are not changed.

        Parameters
        ----------
        n : int
            Index of the asphere coefficient, starting at 1 for the coefficient of :math:`r^2`. Index 2 is the
            coefficient of :math:`r^4`, etc., up to index 8 for :math:`r^{16}`.
        value : float
            The value of the asphere coefficient.

        Raises
        ------
        ValueError
            If `n` is smaller than 1 or larger than 8.
        RuntimeError
            If the surface has not been built.
        """
        self._validate_coefficient(n)

        self._surface_data.SetNthEvenOrderTerm(n, value)

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        surface_index = super().build(optical_system, position=position, replace_existing=replace_existing)

        for n, value in enumerate(self._asphere_coefficients, start=1):
            self.set_asphere_coefficient(n, value)

        return surface_index


class BaseOpticStudioZernikeSurface(OpticStudioSurface, ABC):
    """Base class for Zernike surfaces in OpticStudio.

    This class provides methods and properties shared by all Zernike surfaces.
    """

    def __new__(cls, *args, **kwargs):  # ruff: ignore[unused-static-method-argument]
        if cls is BaseOpticStudioZernikeSurface:
            raise TypeError("Only child classes of BaseOpticStudioZernikeSurface may be instantiated.")

        return super().__new__(cls)

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
        zernike_coefficients: ZernikeCoefficients | None = None,
    ):
        super().__init__(
            comment=comment,
            radius=radius,
            thickness=thickness,
            semi_diameter=semi_diameter,
            conic=conic,
            material=material,
            is_stop=is_stop,
        )

        if zernike_coefficients is not None:
            if any(key > number_of_terms for key in zernike_coefficients):
                raise ValueError(
                    f"Zernike coefficients must be less than or equal to the maximum term {number_of_terms}."
                )
            if any(key < 1 for key in zernike_coefficients):
                raise ValueError("Zernike coefficients must be larger than 0.")

        self._number_of_terms = number_of_terms
        self._norm_radius = norm_radius
        self._zernike_coefficients = zernike_coefficients or ZernikeCoefficients(
            dict.fromkeys(range(1, number_of_terms + 1), 0)
        )

    number_of_terms: OpticStudioSurfaceDataProperty[int] = OpticStudioSurfaceDataProperty("NumberOfTerms")
    norm_radius: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("NormRadius")

    def _validate_coefficient(self, n: int):
        if n < 1:
            raise ValueError("Zernike coefficient must be larger than 0.")
        number_of_terms = self.number_of_terms
        if number_of_terms is None:
            raise RuntimeError("The surface has not been built.")
        if n > number_of_terms:
            raise ValueError(f"Zernike coefficient must be smaller than the maximum term {number_of_terms}.")

    def get_zernike_coefficient(self, n: int) -> float:
        """Get the value of the nth Zernike coefficient.

        Parameters
        ----------
        n : int
            The Zernike coefficient to retrieve.

        Returns
        -------
        float
            The value of the Zernike coefficient.
        """
        self._validate_coefficient(n)

        return self.surface.SurfaceData.GetNthZernikeCoefficient(n)

    def set_zernike_coefficient(self, n: int, value: float) -> None:
        """Set the value of the nth Zernike coefficient.

        Parameters
        ----------
        n : int
            The Zernike coefficient to set.
        value : float
            The value of the Zernike coefficient.
        """
        self._validate_coefficient(n)

        self.surface.SurfaceData.SetNthZernikeCoefficient(n, value)

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        index = super().build(optical_system, position=position, replace_existing=replace_existing)

        self.number_of_terms = self._number_of_terms
        self.norm_radius = self._norm_radius

        for n, value in self._zernike_coefficients.items():
            self.set_zernike_coefficient(n, value)

        return index


class OpticStudioZernikeStandardSagSurface(BaseOpticStudioZernikeSurface):
    """Zernike Standard Sag surface in OpticStudio."""

    _TYPE = "ZernikeStandardSag"

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
        extrapolate: int = 0,
        zernike_decenter_x: float = 0.0,
        zernike_decenter_y: float = 0.0,
        number_of_terms: int = 0,
        norm_radius: float = 100,
        zernike_coefficients: ZernikeCoefficients | None = None,
    ):
        super().__init__(
            comment=comment,
            radius=radius,
            thickness=thickness,
            semi_diameter=semi_diameter,
            conic=conic,
            material=material,
            is_stop=is_stop,
            number_of_terms=number_of_terms,
            norm_radius=norm_radius,
            zernike_coefficients=zernike_coefficients,
        )

        self._extrapolate = extrapolate
        self._zernike_decenter_x = zernike_decenter_x
        self._zernike_decenter_y = zernike_decenter_y

    extrapolate: OpticStudioSurfaceDataProperty[int] = OpticStudioSurfaceDataProperty("Extrapolate")
    zernike_decenter_x: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("ZernikeDecenter_X")
    zernike_decenter_y: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("ZernikeDecenter_Y")

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        index = super().build(optical_system, position=position, replace_existing=replace_existing)

        self.extrapolate = self._extrapolate
        self.zernike_decenter_x = self._zernike_decenter_x
        self.zernike_decenter_y = self._zernike_decenter_y

        return index


class OpticStudioZernikeStandardPhaseSurface(BaseOpticStudioZernikeSurface):
    """Zernike Standard Phase surface in OpticStudio."""

    _TYPE = "ZernikeStandardPhase"

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
        extrapolate: int = 0,
        diffract_order: float = 0.0,
        number_of_terms: int = 0,
        norm_radius: float = 100,
        zernike_coefficients: ZernikeCoefficients | None = None,
    ):
        super().__init__(
            comment=comment,
            radius=radius,
            thickness=thickness,
            semi_diameter=semi_diameter,
            conic=conic,
            material=material,
            is_stop=is_stop,
            number_of_terms=number_of_terms,
            norm_radius=norm_radius,
            zernike_coefficients=zernike_coefficients,
        )

        self._extrapolate = extrapolate
        self._diffract_order = diffract_order

    extrapolate: OpticStudioSurfaceDataProperty[int] = OpticStudioSurfaceDataProperty("Extrapolate")
    diffract_order: OpticStudioSurfaceDataProperty[float] = OpticStudioSurfaceDataProperty("DiffractOrder")

    def build(self, optical_system: OpticStudioSystem, *, position: int, replace_existing: bool = False) -> int:
        """Create the surface in OpticStudio.

        Create the surface in the provided `OpticStudioSystem` at index `position`.
        By default, a new surface will be created. An existing surface will be overwritten if `replace_existing`
        is set to `True`.

        Parameters
        ----------
        optical_system : zospy.zpcore.OpticStudioSystem
            OpticStudio system in which the surface is created.
        position : int
            Index at which the surface is located, starting at 0 for the object surface.
        replace_existing : bool
            If `True`, replace an existing surface instead of inserting a new one. Defaults to `False`.

        Returns
        -------
        int
            The index of the created surface. Subsequent surfaces should be added after this index.
        """
        index = super().build(optical_system, position=position, replace_existing=replace_existing)

        self.extrapolate = self._extrapolate
        self.diffract_order = self._diffract_order

        return index


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
    material: str | MaterialModel,
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
def _make_surface(
    surface: Stop,
    material: str | MaterialModel = "",
    comment: str = "",
) -> OpticStudioSurface:
    return OpticStudioSurface(
        comment=comment,
        thickness=surface.thickness,
        material=material,
        semi_diameter=surface.semi_diameter,
        is_stop=True,
    )


@make_surface.register
def _make_surface(
    surface: BiconicSurface,
    material: str | MaterialModel = "",
    comment: str = "",
) -> OpticStudioBiconicSurface:
    return OpticStudioBiconicSurface(
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
    surface: EvenAsphereSurface,
    material: str | MaterialModel = "",
    comment: str = "",
) -> OpticStudioEvenAsphereSurface:
    return OpticStudioEvenAsphereSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        asphere_coefficients=surface.asphere_coefficients,
        material=material,
        # Setting is_stop to False is not supported in OpticStudio
        is_stop=surface.is_stop or None,
    )


@make_surface.register
def _make_surface(
    surface: ZernikeStandardSagSurface,
    material: str | MaterialModel = "",
    comment: str = "",
) -> OpticStudioZernikeStandardSagSurface:
    return OpticStudioZernikeStandardSagSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        material=material,
        zernike_coefficients=ZernikeCoefficients(surface.zernike_coefficients),
        extrapolate=1 if surface.extrapolate else 0,
        zernike_decenter_x=surface.zernike_decenter_x,
        zernike_decenter_y=surface.zernike_decenter_y,
        number_of_terms=surface.maximum_term or 0,
        norm_radius=surface.norm_radius,
    )


@make_surface.register
def _make_surface(
    surface: ZernikeStandardPhaseSurface,
    material: str | MaterialModel = "",
    comment: str = "",
) -> OpticStudioZernikeStandardPhaseSurface:
    return OpticStudioZernikeStandardPhaseSurface(
        comment=comment,
        radius=surface.radius,
        thickness=surface.thickness,
        semi_diameter=surface.semi_diameter,
        conic=surface.asphericity,
        material=material,
        zernike_coefficients=ZernikeCoefficients(surface.zernike_coefficients),
        extrapolate=1 if surface.extrapolate else 0,
        diffract_order=surface.diffraction_order,
        number_of_terms=surface.maximum_term or 0,
        norm_radius=surface.norm_radius,
    )


@make_surface.register  # pyrefly: ignore[no-matching-overload]
def _make_surface(
    surface: NoSurface,  # ruff: ignore[unused-function-argument]
    material: None = None,  # ruff: ignore[unused-function-argument]
    comment: str = "",  # ruff: ignore[unused-function-argument]
) -> OpticStudioNoSurface:
    return OpticStudioNoSurface()
