"""Models for the ocular geometry."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, fields
from functools import cache
from sys import version_info
from typing import Any, Generic, NamedTuple, cast

import numpy as np

from visisipy.models.helpers import _collect_subclasses
from visisipy.types import TypedDict
from visisipy.wavefront import ZernikeCoefficients

# Use typing_extensions.TypeVar for Python <3.13 to support default values for type variables
if version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar

__all__ = (
    "BiconicSurface",
    "EyeGeometry",
    "NoSurface",
    "StandardSurface",
    "Stop",
    "Surface",
    "ZernikeStandardPhaseSurface",
    "ZernikeStandardSagSurface",
)


@dataclass
class Surface(ABC):  # noqa: B024
    """Base class for optical surfaces.

    Attributes
    ----------
    thickness : float
        The thickness of the surface. Default is 0.
    """

    thickness: float = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the surface to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the surface, including a ``"type"`` key with the class name.
        """
        result: dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):
            result[f.name] = getattr(self, f.name)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Surface:
        """Create a surface from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary with the surface parameters. Must contain a ``"type"`` key with the class name.

        Returns
        -------
        Surface
            A surface instance of the class specified by ``data["type"]``.

        Raises
        ------
        ValueError
            If the ``"type"`` value in ``data`` does not correspond to a known surface class.
        """
        data = dict(data)
        type_name = data.pop("type", cls.__name__)
        registry = _get_surface_registry()
        target_cls = registry.get(type_name)
        if target_cls is None:
            msg = f"Unknown surface type: {type_name!r}"
            raise ValueError(msg)
        init_field_names = {f.name for f in fields(target_cls) if f.init}
        kwargs = {k: v for k, v in data.items() if k in init_field_names}
        return target_cls(**kwargs)


class _EllipsoidRadii(NamedTuple):
    z: float
    y: float
    x: float

    @property
    def anterior_posterior(self) -> float:
        """Anterior-posterior radius of the ellipsoid."""
        return self.z

    @property
    def inferior_superior(self) -> float:
        """Inferior-superior radius of the ellipsoid."""
        return self.y

    @property
    def left_right(self) -> float:
        """Left-right radius of the ellipsoid.

        This is the left-right direction as seen from the anatomical position. For the left eye,
        this corresponds to the temporal-nasal direction, while for the right eye, it corresponds to
        the nasal-temporal direction.
        """
        return self.x


@dataclass
class StandardSurface(Surface):
    """Standard conic surface.

    Represents surfaces as conic sections (spheres, ellipsoids, paraboloids, hyperboloids), defined in terms of their
    radius of curvature and asphericity. For the asphericity, the following definition is used:

    .. math::
        k = - \varepsilon^2 = - \\left( 1 - \frac{b^2}{a^2} \right)

    with :math:`k` the asphericity, :math:`\varepsilon` the eccentricity, :math:`a` the ellipsoid axis parallel to
    the optical axis and :math:`b` the ellipsoid axis perpendicular to the optical axis. This is the definition used
    in OpticStudio. Note that the meaning of :math:`a` and :math:`b` differs from the standard definition of an ellipse,
    where they are the semi-major and semi-minor axes.

    Attributes
    ----------
    radius : float
        The radius of the surface. Default is infinity.
    asphericity : float
        The asphericity of the surface. Default is 0.
    thickness : float
        The thickness of the surface. Default is 0.
    semi_diameter : float | None
        The semi-diameter of the surface aperture. Default is `None`.
    is_stop : bool
        If `True`, the surface is a stop surface. Default is `False`.

    Methods
    -------
    ellipsoid_radii(self) -> tuple[float, float, float]:
        Calculates and returns the ellipsoid x, y and z radii (semi-axes) of the surface.
    """

    radius: float = float("inf")
    asphericity: float = 0
    thickness: float = 0
    semi_diameter: float | None = None
    is_stop: bool = False

    @property
    def ellipsoid_radii(self) -> _EllipsoidRadii:
        """Calculates and returns the ellipsoid radii (semi-axes) of the surface.

        This works only if the surface is an ellipsoid (asphericity > -1), otherwise a NotImplementedError is raised.
        A tuple of the radii along the z, y and x axes is returned, where the z axis is the optical axis.
        These axes correspond to the following anatomical directions:

        - z: anterior-posterior
        - y: inferior-superior
        - x: left-right

        Returns
        -------
        tuple[float, float, float]
            The ellipsoid radii (semi-axes) of the surface.

        Raises
        ------
        NotImplementedError
            If the surface is not an ellipsoid (asphericity <= -1).
        """
        if self.asphericity > -1:
            axial = self.radius / (self.asphericity + 1)
            radial = abs(self.radius / np.sqrt(self.asphericity + 1))
            return _EllipsoidRadii(
                z=axial,
                y=radial,
                x=radial,
            )

        raise NotImplementedError(
            f"Ellipsoid radii are only defined for ellipsoids (asphericity > -1), got {self.asphericity=}"
        )


@dataclass
class Stop(StandardSurface):
    """Stop surface.

    This surface represents the aperture stop of an optical system.

    Attributes
    ----------
    semi_diameter : float
        The semi-diameter of the aperture stop. Default is 1.
    thickness : float
        The thickness of the stop. Default is 0.
    """

    semi_diameter: float = 1.0
    thickness: float = 0
    is_stop: bool = True


@dataclass
class BiconicSurface(StandardSurface):
    """Standard biconic surface.

    Inherits from the `StandardSurface` class and represents a surface with different radii of curvature and
    asphericities in the x (left-right) and y (inferior-superior) directions. This is useful for modeling astigmatic surfaces.
    For the left eye, the x (left-right) direction corresponds to the temporal-nasal direction, while for the right eye,
    it corresponds to the nasal-temporal direction.

    Attributes
    ----------
    radius : float
        The radius of the surface in the y (inferior-superior) direction. Default is infinity.
    radius_x : float
        The radius of the surface in the x (left-right) direction. Default is infinity.
    asphericity : float
        The asphericity of the surface in the y (inferior-superior) direction. Default is 0.
    asphericity_x : float
        The asphericity of the surface in the x (left-right) direction. Default is 0.
    thickness : float
        The thickness of the surface. Default is 0.
    semi_diameter : float | None
        The semi-diameter of the surface aperture. Default is `None`.
    is_stop : bool
        If `True`, the surface is a stop surface. Default is `False`.

    Methods
    -------
    ellipsoid_radii(self) -> tuple[float, float, float]:
        Calculates and returns the ellipsoid x, y and z radii of the surface.
    """

    radius_x: float = float("inf")
    asphericity_x: float = 0

    @property
    def ellipsoid_radii(self) -> _EllipsoidRadii:
        """Calculates and returns the ellipsoid radii (semi-axes) of the surface.

        This works only if the surface is an ellipsoid (asphericity > -1), otherwise a NotImplementedError is raised.
        A tuple of the radii along the z, y and x axes is returned, where the z axis is the optical axis.
        These axes correspond to the following anatomical directions:

        - z: anterior-posterior
        - y: inferior-superior
        - x: left-right

        Returns
        -------
        tuple[float, float, float]
            The ellipsoid radii (semi-axes) of the surface.

        Raises
        ------
        NotImplementedError
            If the surface is not an ellipsoid (asphericity <= -1).
        """
        if self.asphericity <= -1 or self.asphericity_x <= -1:
            raise NotImplementedError(
                f"Half axes are only defined for ellipsoids (asphericity > -1), got {self.asphericity=} and {self.asphericity_x=}"
            )

        # Z radii may differ in the sagittal and tangential planes. If they are not equal, the surface is not an ellipsoid.
        z_radius_x = self.radius_x / (self.asphericity_x + 1)
        z_radius_y = self.radius / (self.asphericity + 1)

        if z_radius_x != z_radius_y:
            raise NotImplementedError(
                "Half axes are only defined for ellipsoids. This biconic surface is not an ellipsoid."
            )

        x_radius = self.radius_x / np.sqrt(self.asphericity_x + 1)
        y_radius = self.radius / np.sqrt(self.asphericity + 1)
        z_radius = abs(z_radius_y)

        return _EllipsoidRadii(
            z=z_radius,
            y=y_radius,
            x=x_radius,
        )


@dataclass
class BaseZernikeStandardSurface(StandardSurface, ABC):
    def __new__(cls, *args, **kwargs):  # noqa: ARG004, RUF100
        if cls == BaseZernikeStandardSurface:
            raise TypeError("Cannot instantiate abstract class BaseZernikeStandardSurface.")
        return super().__new__(cls)

    zernike_coefficients: ZernikeCoefficients | dict[int, float] = field(default_factory=dict)
    maximum_term: int | None = None

    def __post_init__(self):
        if self.maximum_term is None:
            self.maximum_term = max(self.zernike_coefficients.keys(), default=0)

        if any(key > self.maximum_term for key in self.zernike_coefficients):
            raise ValueError("The Zernike coefficients contain terms that are greater than the maximum term.")

        self.zernike_coefficients = ZernikeCoefficients(self.zernike_coefficients)

    def to_dict(self) -> dict[str, Any]:
        """Convert the surface to a dictionary.

        Overrides :meth:`Surface.to_dict` to serialize :class:`~visisipy.wavefront.ZernikeCoefficients`
        as a plain dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the surface, including a ``"type"`` key with the class name.
        """
        result = super().to_dict()
        result["zernike_coefficients"] = dict(self.zernike_coefficients)
        return result


@dataclass
class ZernikeStandardSagSurface(BaseZernikeStandardSurface):
    """Zernike standard coefficients surface with surface deformations.

    Represents a surface with surface deformations described by Zernike polynomials.

    Attributes
    ----------
    zernike_coefficients : ZernikeCoefficients | dict[int, float]
        The Zernike coefficients of the surface. Default is an empty dictionary.
    extrapolate : bool
        If True, the Zernike coefficients will be considered even if the ray lands beyond the normalization radius.
        Default is `True`.
    zernike_decenter_x : float
        Decentration of the Zernike terms with respect to the conical and aspherical terms in the x-direction.
        Default is 0.
    zernike_decenter_y : float
        Decentration of the Zernike terms with respect to the conical and aspherical terms in the y-direction.
        Default is 0.
    maximum_term : int | None
        The maximum Zernike term to consider. If `None` is passed, this will be set to the maximum term in `zernike_coefficients`.
    norm_radius : float
        The normalization radius for the Zernike coefficients, in lens units (usually mm). Default is 100.

    Raises
    ------
    ValueError
        If the Zernike coefficients contain terms that are greater than the maximum term.
    """

    zernike_coefficients: ZernikeCoefficients | dict[int, float] = field(default_factory=dict)
    extrapolate: bool = True
    zernike_decenter_x: float = 0
    zernike_decenter_y: float = 0
    maximum_term: int | None = None
    norm_radius: float = 100


@dataclass
class ZernikeStandardPhaseSurface(BaseZernikeStandardSurface):
    """Zernike standard coefficients surface with wavefront aberrations.

    Represents a surface with wavefront aberrations described by Zernike polynomials.

    Attributes
    ----------
    zernike_coefficients : ZernikeCoefficients | dict[int, float]
        The Zernike coefficients of the surface. Default is an empty dictionary.
    extrapolate : bool
        If True, the Zernike coefficients will be considered even if the ray lands beyond the normalization radius.
        Default is `True`.
    diffraction_order : float
        The diffraction order of the surface. Default is 0.
    maximum_term : int | None
        The maximum Zernike term to consider. If `None` is passed, this will be set to the maximum term in `zernike_coefficients`.
    norm_radius : float
        The normalization radius for the Zernike coefficients, in lens units (usually mm). Default is 100.

    Raises
    ------
    ValueError
        If the Zernike coefficients contain terms that are greater than the maximum term.
    """

    zernike_coefficients: ZernikeCoefficients | dict[int, float] = field(default_factory=dict)
    extrapolate: bool = True
    diffraction_order: float = 1
    maximum_term: int | None = None
    norm_radius: float = 100


@dataclass
class NoSurface(Surface):
    """A surface that does not exist.

    This surface is used to indicate that a surface is not present in the optical system.
    It can be used to define three-surface schematic eyes and reduced eye models.

    .. note::
       This surface does not modify the optical system, i.e. surfaces of the `NoSurface` type are not built
       when calling `EyeModel.build`. This means the properties of the preceding surface, e.g. the refractive index,
       will propagate to the next surface.
    """

    # Thickness is present on all surface instances, but cannot be set from the constructor.
    thickness: int = field(default=0, init=False)


_CorneaFront = TypeVar("_CorneaFront", bound=Surface, default=StandardSurface)
_CorneaBack = TypeVar("_CorneaBack", bound=Surface, default=StandardSurface)
_Pupil = TypeVar("_Pupil", bound=Surface, default=Stop)
_LensFront = TypeVar("_LensFront", bound=Surface, default=StandardSurface)
_LensBack = TypeVar("_LensBack", bound=Surface, default=StandardSurface)
_Retina = TypeVar("_Retina", bound=StandardSurface, default=StandardSurface)


class EyeModelSurfaces(TypedDict, total=False):
    """Surfaces of an eye model."""

    cornea_front: StandardSurface
    cornea_back: StandardSurface
    pupil: Stop
    lens_front: StandardSurface
    lens_back: StandardSurface
    retina: StandardSurface


class EyeGeometry(Generic[_CorneaFront, _CorneaBack, _Pupil, _LensFront, _LensBack, _Retina]):  # noqa: PLW1641
    """Geometric parameters of an eye.

    Sizes are specified in mm. This class is mainly intended as a base class for more specific eye models.
    When used directly, all surfaces need to be specified manually.

    Attributes
    ----------
    cornea_front : Surface
        The front surface of the cornea.
    cornea_back : Surface
        The back surface of the cornea.
    pupil : Surface
        The pupil of the eye.
    lens_front : Surface
        The front surface of the lens.
    lens_back : Surface
        The back surface of the lens.
    retina : Surface
        The retina of the eye.

    Methods
    -------
    axial_length(self) -> float:
        Calculates and returns the axial length of the eye.
    cornea_thickness(self) -> float:
        Returns the thickness of the cornea.
    anterior_chamber_depth(self) -> float:
        Returns the depth of the anterior chamber.
    lens_thickness(self) -> float:
        Returns the thickness of the lens.
    vitreous_thickness(self) -> float:
        Returns the thickness of the vitreous.
    """

    def __init__(
        self,
        cornea_front: _CorneaFront | None = None,
        cornea_back: _CorneaBack | None = None,
        pupil: _Pupil | None = None,
        lens_front: _LensFront | None = None,
        lens_back: _LensBack | None = None,
        retina: _Retina | None = None,
    ) -> None:
        """Initializes an instance of the EyeGeometry class.

        This creates an instance of the EyeGeometry class with the specified surfaces. If a surface is not provided,
        the default values of `StandardSurface` and `Stop` will be used. This class is mainly intended as a base
        class, and instantiating it without specifying the surfaces will not result in a realistic eye model. To
        initialize a realistic eye model, use a subclass such as `NavarroGeometry`.

        Parameters
        ----------
        cornea_front : Surface, optional
            The front surface of the cornea. If not provided, a default Surface instance will be used.
        cornea_back : Surface, optional
            The back surface of the cornea. If not provided, a default Surface instance will be used.
        pupil : Surface, optional
            The pupil of the eye. If not provided, a default Stop instance will be used.
        lens_front : Surface, optional
            The front surface of the lens. If not provided, a default Surface instance will be used.
        lens_back : Surface, optional
            The back surface of the lens. If not provided, a default Surface instance will be used.
        retina : StandardSurface, optional
            The retina of the eye. If not provided, a default Surface instance will be used.
        """

        self.cornea_front = cornea_front or StandardSurface()
        self.cornea_back = cornea_back or StandardSurface()
        self.pupil = pupil or Stop()
        self.lens_front = lens_front or StandardSurface()
        self.lens_back = lens_back or StandardSurface()
        self.retina = retina or StandardSurface()

    @property
    def cornea_front(self) -> _CorneaFront:
        """Cornea front geometry."""
        return self._cornea_front

    @cornea_front.setter
    def cornea_front(self, value: _CorneaFront) -> None:
        self._cornea_front = value

    @property
    def cornea_back(self) -> _CorneaBack:
        """Cornea back geometry."""
        return self._cornea_back

    @cornea_back.setter
    def cornea_back(self, value: _CorneaBack) -> None:
        self._cornea_back = value

    @property
    def pupil(self) -> _Pupil:
        """Pupil geometry."""
        return self._pupil

    @pupil.setter
    def pupil(self, value: _Pupil) -> None:
        if not value.is_stop:
            raise ValueError("The pupil surface must be a stop.")

        self._pupil = value

    @property
    def lens_front(self) -> _LensFront:
        """Lens front geometry."""
        return self._lens_front

    @lens_front.setter
    def lens_front(self, value: _LensFront) -> None:
        self._lens_front = value

    @property
    def lens_back(self) -> _LensBack:
        """Lens back geometry."""
        return self._lens_back

    @lens_back.setter
    def lens_back(self, value: _LensBack) -> None:
        self._lens_back = value

    @property
    def retina(self) -> _Retina:
        """Retina geometry."""
        return self._retina

    @retina.setter
    def retina(self, value: _Retina) -> None:
        if value.asphericity <= -1:
            raise ValueError(f"Only an elliptical retina is allowed (asphericity > -1), got {value.asphericity=}")

        self._retina = value

    @property
    def axial_length(self) -> float:
        """Axial length of the eye, in mm."""
        return (
            self.cornea_front.thickness
            + self.cornea_back.thickness
            + self.pupil.thickness
            + self.lens_front.thickness
            + self.lens_back.thickness
        )

    @property
    def cornea_thickness(self) -> float:
        """Central thickness of the cornea, in mm.

        The corneal thickness is defined as the distance between the anterior and posterior cornea surfaces.
        """
        return self.cornea_front.thickness

    @property
    def anterior_chamber_depth(self) -> float:
        """Depth of the anterior chamber, in mm.

        The anterior chamber depth is defined as the distance between the posterior cornea surface and the anterior lens surface.
        """
        return self.cornea_back.thickness + self.pupil.thickness

    @property
    def aqueous_depth(self) -> float:
        """Depth of the aqueous humor, in mm.

        The aqueous depth is defined as the distance between the posterior cornea surface and the anterior lens surface.

        See Also
        --------
        anterior_chamber_depth : Equivalent to the anterior chamber depth.
        """
        return self.anterior_chamber_depth

    @property
    def pupil_lens_distance(self) -> float:
        """Distance between the pupil and the anterior lens surface, in mm."""
        return self.pupil.thickness

    @property
    def lens_thickness(self) -> float:
        """Thickness of the crystalline lens, in mm.

        The lens thickness is defined as the distance between the anterior and posterior lens surfaces.
        """
        return self.lens_front.thickness

    @property
    def vitreous_thickness(self) -> float:
        """Thickness of the vitreous body, in mm.

        The vitreous thickness is defined as the distance between the posterior lens surface and the retina.
        """
        return self.lens_back.thickness

    def reverse(self):
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert the eye geometry to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the eye geometry, including a ``"type"`` key with the class name
            and the serialized surfaces.
        """
        return {
            "type": type(self).__name__,
            "cornea_front": self.cornea_front.to_dict(),
            "cornea_back": self.cornea_back.to_dict(),
            "pupil": self.pupil.to_dict(),
            "lens_front": self.lens_front.to_dict(),
            "lens_back": self.lens_back.to_dict(),
            "retina": self.retina.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> EyeGeometry[Surface, Surface, Surface, Surface, Surface, StandardSurface]:
        """Create an eye geometry from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary with the eye geometry parameters, as produced by :meth:`to_dict`.

        Returns
        -------
        EyeGeometry
            An :class:`EyeGeometry` instance with surfaces reconstructed from ``data``.

        Raises
        ------
        ValueError
            If the ``"type"`` value in ``data`` does not correspond to a known eye geometry class.
        """
        data = dict(data)

        type_name = data.pop("type", cls.__name__)
        registry = _get_eye_geometry_registry()
        target_cls = registry.get(type_name)
        if target_cls is None:
            msg = f"Unknown geometry type: {type_name!r}"
            raise ValueError(msg)

        return target_cls(
            cornea_front=Surface.from_dict(data["cornea_front"]),
            cornea_back=Surface.from_dict(data["cornea_back"]),
            pupil=Surface.from_dict(data["pupil"]),
            lens_front=Surface.from_dict(data["lens_front"]),
            lens_back=Surface.from_dict(data["lens_back"]),
            retina=cast("StandardSurface", Surface.from_dict(data["retina"])),
        )

    def __repr__(self) -> str:
        name = self.__class__.__name__
        surfaces = {
            "cornea_front": self.cornea_front,
            "cornea_back": self.cornea_back,
            "pupil": self.pupil,
            "lens_front": self.lens_front,
            "lens_back": self.lens_back,
            "retina": self.retina,
        }

        return f"{name}({', '.join(f'{k}={v!r}' for k, v in surfaces.items())})"

    def __key(self):
        """Helper method to generate a tuple of the eye geometry's attributes for hashing and equality checks.

        Returns
        -------
        tuple
            A tuple containing the eye geometry's attributes in a consistent order.
        """
        return (
            self.cornea_front,
            self.cornea_back,
            self.pupil,
            self.lens_front,
            self.lens_back,
            self.retina,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EyeGeometry):
            return NotImplemented

        return self.__key() == other.__key()


@cache
def _get_surface_registry() -> dict[str, type[Surface]]:
    """Build a registry of all :class:`Surface` subclasses by recursively collecting ``__subclasses__``.

    Returns
    -------
    dict[str, type[Surface]]
        Mapping from class name to class for :class:`Surface` and all its subclasses.
    """
    registry: dict[str, type[Surface]] = {}
    _collect_subclasses(Surface, registry)
    return registry


@cache
def _get_eye_geometry_registry() -> dict[
    str, type[EyeGeometry[Surface, Surface, Surface, Surface, Surface, StandardSurface]]
]:
    """Build a registry of all :class:`EyeGeometry` subclasses by recursively collecting ``__subclasses__``.

    Returns
    -------
    dict[str, type[EyeGeometry]]
        Mapping from class name to class for :class:`EyeGeometry` and all its subclasses.
    """
    registry: dict[str, type[EyeGeometry[Surface, Surface, Surface, Surface, Surface, StandardSurface]]] = {}
    _collect_subclasses(EyeGeometry, registry)
    return registry
