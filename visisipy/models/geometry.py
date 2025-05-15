"""Models for the ocular geometry."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeVar
from warnings import warn

import numpy as np

from visisipy.wavefront import ZernikeCoefficients

__all__ = (
    "EyeGeometry",
    "NavarroGeometry",
    "NoSurface",
    "StandardSurface",
    "Stop",
    "Surface",
    "create_geometry",
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


class _HalfAxes2D(NamedTuple):
    axial: float
    radial: float


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

    Methods
    -------
    half_axes(self) -> tuple[float, float]:
        Calculates and returns the half axes of the surface.
    """

    radius: float = float("inf")
    asphericity: float = 0
    thickness: float = 0
    semi_diameter: float | None = None
    is_stop: bool = False

    @property
    def half_axes(self) -> _HalfAxes2D:
        """Calculates and returns the half axes of the surface.

        This works only if the surface is an ellipsoid (asphericity > -1), otherwise a NotImplementedError is raised.
        A tuple of the axial and radial half axes is returned, with the axial axis being parallel to the optical axis
        and the radial axis perpendicular to it.

        Returns
        -------
        tuple[float, float]
            The axial and radial half axes of the surface.

        Raises
        ------
        NotImplementedError
            If the surface is not an ellipsoid (asphericity <= -1).
        """
        if self.asphericity > -1:
            return _HalfAxes2D(
                axial=(self.radius / (self.asphericity + 1)),
                radial=abs(self.radius / np.sqrt(self.asphericity + 1)),
            )

        raise NotImplementedError(
            f"Half axes are only defined for ellipses (asphericity > -1), got {self.asphericity=}"
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
class ZernikeStandardSagSurface(StandardSurface):
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
    maximum_term : int
        The maximum Zernike term to consider. Default is 0.
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
    maximum_term: int = 0
    norm_radius: float = 100

    def __post_init__(self):
        if any(key > self.maximum_term for key in self.zernike_coefficients):
            raise ValueError("The Zernike coefficients contain terms that are greater than the maximum term.")

        self.zernike_coefficients = ZernikeCoefficients(self.zernike_coefficients)


@dataclass
class ZernikeStandardPhaseSurface(StandardSurface):
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
    maximum_term : int
        The maximum Zernike term to consider. Default is 0.
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
    maximum_term: int = 0
    norm_radius: float = 100

    def __post_init__(self):
        if any(key > self.maximum_term for key in self.zernike_coefficients):
            raise ValueError("The Zernike coefficients contain terms that are greater than the maximum term.")

        self.zernike_coefficients = ZernikeCoefficients(self.zernike_coefficients)


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


class EyeGeometry:
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
        cornea_front: StandardSurface | None = None,
        cornea_back: StandardSurface | None = None,
        pupil: Stop | None = None,
        lens_front: StandardSurface | None = None,
        lens_back: StandardSurface | None = None,
        retina: StandardSurface | None = None,
    ):
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
    def cornea_front(self) -> StandardSurface:
        """Cornea front geometry."""
        return self._cornea_front

    @cornea_front.setter
    def cornea_front(self, value: StandardSurface) -> None:
        self._cornea_front = value

    @property
    def cornea_back(self) -> StandardSurface:
        """Cornea back geometry."""
        return self._cornea_back

    @cornea_back.setter
    def cornea_back(self, value: StandardSurface) -> None:
        self._cornea_back = value

    @property
    def pupil(self) -> Stop:
        """Pupil geometry."""
        return self._pupil

    @pupil.setter
    def pupil(self, value: Stop) -> None:
        if not value.is_stop:
            raise ValueError("The pupil surface must be a stop.")

        self._pupil = value

    @property
    def lens_front(self) -> StandardSurface:
        """Lens front geometry."""
        return self._lens_front

    @lens_front.setter
    def lens_front(self, value: StandardSurface) -> None:
        self._lens_front = value

    @property
    def lens_back(self) -> StandardSurface:
        """Lens back geometry."""
        return self._lens_back

    @lens_back.setter
    def lens_back(self, value: StandardSurface) -> None:
        self._lens_back = value

    @property
    def retina(self) -> StandardSurface:
        """Retina geometry."""
        return self._retina

    @retina.setter
    def retina(self, value: StandardSurface) -> None:
        if value.asphericity <= -1:
            raise ValueError(f"Only an elliptical retina is allowed (asphericity > -1), got {value.asphericity=}")

        self._retina = value

    @property
    def axial_length(self) -> float:
        """Axial length of the eye, in mm."""
        return (
            self.cornea_front.thickness
            + self.cornea_back.thickness
            + self.lens_front.thickness
            + self.lens_back.thickness
        )

    @property
    def cornea_thickness(self) -> float:
        """Thickness of the cornea, in mm."""
        return self.cornea_front.thickness

    @property
    def anterior_chamber_depth(self) -> float:
        """Depth of the anterior chamber, in mm."""
        return self.cornea_back.thickness

    @property
    def lens_thickness(self) -> float:
        """Thickness of the crystalline lens, in mm."""
        return self.lens_front.thickness

    @property
    def vitreous_thickness(self) -> float:
        """Thickness of the vitreous body, in mm."""
        return self.lens_back.thickness

    def reverse(self):
        raise NotImplementedError


class NavarroGeometry(EyeGeometry):
    """Geometric parameters of the Navarro schematic eye.

    This schematic eye is based on the Navarro model as described in [1]_.
    Sizes are specified in mm.

     .. [1] Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model.
            JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881

    Attributes
    ----------
    cornea_front : StandardSurface
        The front surface of the cornea.
    cornea_back : StandardSurface
        The back surface of the cornea.
    pupil : Stop
        The pupil of the eye.
    lens_front : StandardSurface
        The front surface of the lens.
    lens_back : StandardSurface
        The back surface of the lens.
    retina : StandardSurface
        The retina of the eye.

    Examples
    --------
    Use the default Navarro geometry:

    >>> from visisipy import NavarroGeometry
    >>> geometry = NavarroGeometry()

    Create a Navarro geometry with a custom retina:

    >>> geometry = NavarroGeometry(
    ...     retina=StandardSurface(radius=-12.5, asphericity=0.5)
    ... )

    Create a default Navarro gometry and change only the lens back radius:

    >>> geometry = NavarroGeometry()
    >>> geometry.lens_back.radius = -5.8
    """

    def __init__(self, **kwargs):
        surfaces = {
            "cornea_front": StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55),
            "cornea_back": StandardSurface(radius=6.50, asphericity=0, thickness=3.05),
            "pupil": Stop(semi_diameter=1.348),
            "lens_front": StandardSurface(radius=10.2, asphericity=-3.1316, thickness=4.0),
            "lens_back": StandardSurface(radius=-6.0, asphericity=-1, thickness=16.3203),
            "retina": StandardSurface(radius=-12.0, asphericity=0),
        }
        surfaces.update(**kwargs)
        super().__init__(**surfaces)


def _update_attribute_if_specified(obj: Surface, attribute: str, value: Any):
    if value is not None:
        setattr(obj, attribute, value)


def _calculate_vitreous_thickness(
    geometry: EyeGeometry,
    axial_length: float | None = None,
    cornea_thickness: float | None = None,
    anterior_chamber_depth: float | None = None,
    lens_thickness: float | None = None,
) -> float:
    """Calculate the thickness of the vitreous body for a partially initialized eye geometry."""
    _axial_length = geometry.axial_length if axial_length is None else axial_length
    _cornea_thickness = geometry.cornea_thickness if cornea_thickness is None else cornea_thickness
    _anterior_chamber_depth = (
        geometry.anterior_chamber_depth if anterior_chamber_depth is None else anterior_chamber_depth
    )
    _lens_thickness = geometry.lens_thickness if lens_thickness is None else lens_thickness

    if None in {
        _axial_length,
        _cornea_thickness,
        _anterior_chamber_depth,
        _lens_thickness,
    }:
        raise ValueError("Cannot calculate vitreous thickness from the supplied parameters.")

    return _axial_length - (_cornea_thickness + _anterior_chamber_depth + _lens_thickness)


GeometryType = TypeVar("GeometryType", bound=EyeGeometry)


def create_geometry(
    base: type[GeometryType] = NavarroGeometry,
    axial_length: float | None = None,
    cornea_thickness: float | None = None,
    cornea_front_radius: float | None = None,
    cornea_front_asphericity: float | None = None,
    cornea_back_radius: float | None = None,
    cornea_back_asphericity: float | None = None,
    anterior_chamber_depth: float | None = None,
    pupil_radius: float | None = None,
    lens_thickness: float | None = None,
    lens_back_radius: float | None = None,
    lens_back_asphericity: float | None = None,
    lens_front_radius: float | None = None,
    lens_front_asphericity: float | None = None,
    retina_radius: float | None = None,
    retina_asphericity: float | None = None,
    *,
    estimate_cornea_back: bool = False,
) -> GeometryType:
    """Create a geometry instance from clinically used parameters.

    All parameters are optional, and if not provided, the default values will be used.
    Sizes are specified in mm. If `estimate_cornea_back` is True, the back cornea radius will be estimated from the
    front cornea radius as `cornea_back_radius = 0.81 * cornea_front_radius`.

    Parameters
    ----------
    base : type[GeometryType]
        The base geometry class to use. Must be a subclass of EyeGeometry.
    axial_length : float, optional
        Axial length of the eye, measured from cornea front to retina.
    cornea_thickness : float, optional
        Thickness of the cornea.
    cornea_front_radius : float, optional
        Radius of curvature of the frontal cornea surface.
    cornea_front_asphericity : float, optional
        Asphericity of the frontal cornea surface.
    cornea_back_radius : float, optional
        Radius of curvature of the back cornea surface.
    cornea_back_asphericity : float, optional
        Asphericity of the back cornea surface.
    anterior_chamber_depth : float, optional
        Depth of the anterior chamber.
    pupil_radius : float, optional
        Radius of the pupil.
    lens_thickness : float, optional
        Thickness of the crystalline lens.
    lens_back_radius : float, optional
        Radius of curvature of the back lens surface.
    lens_back_asphericity : float, optional
        Asphericity of the back lens surface.
    lens_front_radius : float, optional
        Radius of curvature of the frontal lens surface.
    lens_front_asphericity : float, optional
        Asphericity of the frontal lens surface.
    retina_radius : float, optional
        Radius of curvature of the retina.
    retina_asphericity : float, optional
        Asphericity of the retina.
    estimate_cornea_back : bool, optional
        If True, the back cornea radius will be estimated from the front cornea radius. Default is `False`.

    Returns
    -------
    EyeGeometry
        An instance of the EyeGeometry class.

    Raises
    ------
    ValueError
        If the base geometry is not a class or if it is not a subclass of EyeGeometry.
    """
    if not isinstance(base, type):
        raise TypeError("The base geometry must be a class. Did you put parentheses after the class name?")

    if not issubclass(base, EyeGeometry):
        raise TypeError("The base geometry must be a subclass of EyeGeometry.")

    if estimate_cornea_back and cornea_back_radius is not None:
        warn("The cornea back radius was provided, but it will be ignored because estimate_cornea_back is True.")

    geometry = base()

    _update_attribute_if_specified(geometry.cornea_front, "thickness", cornea_thickness)
    _update_attribute_if_specified(geometry.cornea_front, "radius", cornea_front_radius)
    _update_attribute_if_specified(geometry.cornea_front, "asphericity", cornea_front_asphericity)

    if estimate_cornea_back:
        cornea_back_radius = 0.81 * geometry.cornea_front.radius

    _update_attribute_if_specified(geometry.cornea_back, "thickness", anterior_chamber_depth)
    _update_attribute_if_specified(geometry.cornea_back, "radius", cornea_back_radius)
    _update_attribute_if_specified(geometry.cornea_back, "asphericity", cornea_back_asphericity)

    _update_attribute_if_specified(geometry.pupil, "semi_diameter", pupil_radius)

    _update_attribute_if_specified(geometry.lens_front, "thickness", lens_thickness)
    _update_attribute_if_specified(geometry.lens_front, "radius", lens_front_radius)
    _update_attribute_if_specified(geometry.lens_front, "asphericity", lens_front_asphericity)

    vitreous_thickness = _calculate_vitreous_thickness(
        geometry, axial_length, cornea_thickness, anterior_chamber_depth, lens_thickness
    )

    if vitreous_thickness <= 0:
        raise ValueError(
            "The sum of the cornea thickness, anterior chamber depth and lens thickness is greater than "
            "or equal to the axial length."
        )

    _update_attribute_if_specified(geometry.lens_back, "thickness", vitreous_thickness)
    _update_attribute_if_specified(geometry.lens_back, "radius", lens_back_radius)
    _update_attribute_if_specified(geometry.lens_back, "asphericity", lens_back_asphericity)

    _update_attribute_if_specified(geometry.retina, "radius", retina_radius)
    _update_attribute_if_specified(geometry.retina, "asphericity", retina_asphericity)

    return geometry
