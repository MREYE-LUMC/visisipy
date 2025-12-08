"""Factory functions for creating eye geometries."""

from __future__ import annotations

from typing import Any, Literal, TypeVar
from warnings import warn

from visisipy.models.geometry import EyeGeometry, Surface
from visisipy.models.zoo.navarro import NavarroGeometry
from visisipy.types import TypedDict, Unpack

__all__ = ("create_geometry",)


def _update_attribute_if_specified(obj: Surface, attribute: str, value: Any):
    if value is not None:
        setattr(obj, attribute, value)


def _calculate_vitreous_thickness(geometry: EyeGeometry, parameters: GeometryParameters) -> float:
    """Calculate the thickness of the vitreous body for a partially initialized eye geometry."""

    # Axial length may be undefined, so parameters.get will not work here.
    _axial_length = parameters["axial_length"] if "axial_length" in parameters else geometry.axial_length
    _cornea_thickness = parameters.get("cornea_thickness", geometry.cornea_thickness)
    _anterior_chamber_depth = parameters.get("anterior_chamber_depth", geometry.anterior_chamber_depth)
    _pupil_lens_distance = parameters.get("pupil_lens_distance", geometry.pupil_lens_distance)
    _lens_thickness = parameters.get("lens_thickness", geometry.lens_thickness)

    if None in {
        _axial_length,
        _cornea_thickness,
        _anterior_chamber_depth,
        _pupil_lens_distance,
        _lens_thickness,
    }:
        raise ValueError("Cannot calculate vitreous thickness from the supplied parameters.")

    return _axial_length - (_cornea_thickness + _anterior_chamber_depth + _pupil_lens_distance + _lens_thickness)


def _check_sign(value: float | None, name: str, sign: Literal["+", "-"]) -> None:
    """Check the sign of a value and warn if it does not match the expected sign.

    Parameters
    ----------
    value : float, optional
        The value to check. If `None`, no check is performed.
    name : str
        The name of the parameter.
    sign : Literal["+", "-"]
        The expected sign of the value.

    Raises
    ------
    ValueError
        If an invalid sign is specified.

    Warns
    -----
    UserWarning
        If the value does not have the expected sign.
    """
    if value is None:
        return

    if sign not in {"+", "-"}:
        raise ValueError(f"Invalid sign '{sign}' specified for {name}. Must be '+' or '-'.")

    msg = "Expected a {} value for {}, got {}. Check if the sign is correct."

    if sign == "+" and value < 0:
        warn(msg.format("positive", name, value), UserWarning, stacklevel=2)
    elif sign == "-" and value > 0:
        warn(msg.format("negative", name, value), UserWarning, stacklevel=2)


GeometryType = TypeVar("GeometryType", bound=EyeGeometry)


class GeometryParameters(TypedDict, total=False):
    """Parameters for the geometry of the eye."""

    axial_length: float
    cornea_thickness: float
    cornea_front_radius: float
    cornea_front_asphericity: float
    cornea_back_radius: float
    cornea_back_asphericity: float
    anterior_chamber_depth: float
    pupil_radius: float
    pupil_lens_distance: float
    lens_thickness: float
    lens_back_radius: float
    lens_back_asphericity: float
    lens_front_radius: float
    lens_front_asphericity: float
    retina_radius: float
    retina_asphericity: float
    retina_ellipsoid_z_radius: float
    retina_ellipsoid_y_radius: float


def create_geometry(
    base: type[GeometryType] = NavarroGeometry,
    *,
    estimate_cornea_back: bool = False,
    **parameters: Unpack[GeometryParameters],
) -> GeometryType:
    """Create a geometry instance from clinically used parameters.

    All parameters are optional, and if not provided, the default values will be used.
    Sizes are specified in mm. If `estimate_cornea_back` is True, the back cornea radius will be estimated from the
    front cornea radius as `cornea_back_radius = 0.81 * cornea_front_radius`.
    The retina can be specified either by its radius and asphericity or by its y and z ellipsoid radii. If both
    methods are specified, a ValueError will be raised.

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
    pupil_lens_distance : float, optional
        Distance between the pupil and the lens.
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
    retina_ellipsoid_z_radius : float, optional
        Radius (semi-axis length) of the retina ellipsoid in the z-direction.
    retina_ellipsoid_y_radius : float, optional
        Radius (semi-axis length) of the retina ellipsoid in the y-direction. For rotationally symmetric retinas,
        this is also the radius in the x-direction.
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
        If the retina radius/asphericity and y/z ellipsoid radii are both specified.
        If only one of the retina ellipsoid radii is specified.
        If the sum of the cornea thickness, anterior chamber depth and lens thickness is greater than or equal to the
        axial length.
    """
    if not isinstance(base, type):
        raise TypeError("The base geometry must be a class. Did you put parentheses after the class name?")

    if not issubclass(base, EyeGeometry):
        raise TypeError("The base geometry must be a subclass of EyeGeometry.")

    if estimate_cornea_back and parameters.get("cornea_back_radius") is not None:
        warn("The cornea back radius was provided, but it will be ignored because estimate_cornea_back is True.")

    # Check signs of parameters
    _check_sign(parameters.get("cornea_front_radius"), "cornea_front_radius", "+")
    _check_sign(parameters.get("cornea_back_radius"), "cornea_back_radius", "+")
    _check_sign(parameters.get("lens_front_radius"), "lens_front_radius", "+")
    _check_sign(parameters.get("lens_back_radius"), "lens_back_radius", "-")
    _check_sign(parameters.get("retina_radius"), "retina_radius", "-")
    _check_sign(parameters.get("retina_ellipsoid_z_radius"), "retina_ellipsoid_z_radius", "-")
    _check_sign(parameters.get("retina_ellipsoid_y_radius"), "retina_ellipsoid_y_radius", "+")

    has_retina_radius_or_asphericity = ("retina_radius" in parameters) or ("retina_asphericity" in parameters)
    has_retina_ellipsoid_radii = ("retina_ellipsoid_z_radius" in parameters) or (
        "retina_ellipsoid_y_radius" in parameters
    )

    if has_retina_radius_or_asphericity and has_retina_ellipsoid_radii:
        raise ValueError("Cannot specify both retina radius/asphericity and ellipsoid radii.")

    if has_retina_ellipsoid_radii and not all(
        key in parameters for key in ("retina_ellipsoid_z_radius", "retina_ellipsoid_y_radius")
    ):
        raise ValueError("If the retina ellipsoid radii are specified, both the z and y radius must be provided.")

    geometry = base()

    _update_attribute_if_specified(geometry.cornea_front, "thickness", parameters.get("cornea_thickness"))
    _update_attribute_if_specified(geometry.cornea_front, "radius", parameters.get("cornea_front_radius"))
    _update_attribute_if_specified(geometry.cornea_front, "asphericity", parameters.get("cornea_front_asphericity"))

    if estimate_cornea_back:
        parameters["cornea_back_radius"] = 0.81 * geometry.cornea_front.radius

    _update_attribute_if_specified(geometry.cornea_back, "thickness", parameters.get("anterior_chamber_depth"))
    _update_attribute_if_specified(geometry.cornea_back, "radius", parameters.get("cornea_back_radius"))
    _update_attribute_if_specified(geometry.cornea_back, "asphericity", parameters.get("cornea_back_asphericity"))

    _update_attribute_if_specified(geometry.pupil, "semi_diameter", parameters.get("pupil_radius"))
    _update_attribute_if_specified(geometry.pupil, "thickness", parameters.get("pupil_lens_distance"))

    _update_attribute_if_specified(geometry.lens_front, "thickness", parameters.get("lens_thickness"))
    _update_attribute_if_specified(geometry.lens_front, "radius", parameters.get("lens_front_radius"))
    _update_attribute_if_specified(geometry.lens_front, "asphericity", parameters.get("lens_front_asphericity"))

    vitreous_thickness = _calculate_vitreous_thickness(geometry, parameters)

    if vitreous_thickness <= 0:
        raise ValueError(
            "The sum of the cornea thickness, anterior chamber depth, pupil-lens distance and lens thickness is "
            "greater than or equal to the axial length."
        )

    _update_attribute_if_specified(geometry.lens_back, "thickness", vitreous_thickness)
    _update_attribute_if_specified(geometry.lens_back, "radius", parameters.get("lens_back_radius"))
    _update_attribute_if_specified(geometry.lens_back, "asphericity", parameters.get("lens_back_asphericity"))

    if has_retina_radius_or_asphericity:
        _update_attribute_if_specified(geometry.retina, "radius", parameters.get("retina_radius"))
        _update_attribute_if_specified(geometry.retina, "asphericity", parameters.get("retina_asphericity"))
    elif has_retina_ellipsoid_radii:
        retina_radius = parameters["retina_ellipsoid_y_radius"] ** 2 / parameters["retina_ellipsoid_z_radius"]
        retina_asphericity = (
            parameters["retina_ellipsoid_y_radius"] ** 2 / parameters["retina_ellipsoid_z_radius"] ** 2 - 1
        )
        _update_attribute_if_specified(geometry.retina, "radius", retina_radius)
        _update_attribute_if_specified(geometry.retina, "asphericity", retina_asphericity)

    return geometry
