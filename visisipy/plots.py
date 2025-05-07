"""Visualization tools for Visisipy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from scipy.optimize import fsolve

from visisipy.models import EyeGeometry, EyeModel

if TYPE_CHECKING:
    from matplotlib.axes import Axes


__all__ = ("plot_eye",)


def plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
) -> Path | tuple[Path, float]:
    """Plot a conic surface.

    Creates a MatPlotLib `Path` for a conic segment, intersecting the horizontal axis in `position`.
    The radius of curvature at the apex is specified as `radius`, the asfericity as `conic`.
    The segment is cut off at the x-coordinate specified by `cutoff`.
    The orientation of the surface is controlled with the sign of `radius`.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the surface.

    Arguments
    ---------
    position : float
        Position of the surface at the optical axis.
    radius : float
        Radius of curvature at the intersection with the horizontal axis.
    conic : float
        Conic constant (asphericity) of the surface. The surface is a hyperbola for `conic < -1`, a
        parabola for `conic == -1` and an ellipse for `conic > -1`.
    cutoff : float
        Position coordinate at which the surface is cut off.
    return_endpoint : bool
        If true, returns the y coordinate of the arc endpoint.
    """
    if conic > -1:
        return plot_ellipse(position, radius, conic, cutoff, return_endpoint=return_endpoint)
    if conic == -1:
        return plot_parabola(position, radius, cutoff, return_endpoint=return_endpoint)

    return plot_hyperbola(position, radius, conic, cutoff)


def _get_ellipse_sizes(radius: float, conic: float) -> tuple[float, float]:
    """Calculate rx and ry (axial and radial radii) of an ellipse from its radius of curvature and conic."""
    # Prolate or oblate does not matter in this case
    return radius / (conic + 1), radius / np.sqrt(conic + 1)


def plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
) -> Path | tuple[Path, float]:
    """Plot a segment of an ellipse.

    Creates an `Arc` patch for an ellipse. The radius of curvature at the apex is specified as `radius`.
    The ellipse is cut off at the x-coordinate specified by `cutoff`.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the ellipse segment.
    float
        y-coordinate of the segment's end point.

    Raises
    ------
    ValueError
        If the cutoff is located outside the ellipse.

    Arguments
    ---------
    position : float
        Coordinate of the ellipse apex.
    radius : float
        Radius of curvature at the apex.
    conic : float
        Conic constant (asphericity) of the ellipse. Must be > -1.
    cutoff : float
        x-coordinate at which the ellipse is cut off.
    return_endpoint : bool
        If true, returns the y coordinate of the arc endpoint.
    """
    rx, ry = _get_ellipse_sizes(radius, conic)

    x0 = position + rx

    if abs(cutoff) > abs(rx):
        message = f"cutoff is located outside the ellipse: {cutoff=}, {rx=}"
        raise ValueError(message)

    t_max = np.abs(np.arccos((cutoff - x0) / rx))
    # This is a bit weird, but necessary to draw the arc in the right direction
    t = np.linspace(-t_max, t_max - 2 * np.pi, 1000)

    vertices = list(zip(x0 + rx * np.cos(t), ry * np.sin(t)))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    ellipse = Path(vertices, codes)

    return (ellipse, ry * np.sin(t_max)) if return_endpoint else ellipse


def plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,  # noqa: ARG001
) -> Path:
    """Plot a segment of a parabola.

    Creates a `Path` for a parabola. The radius of curvature at the apex is specified as `radius`.
    The parabola is cut off at the x-coordinate specified by `cutoff`.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the parabola segment.
    float
        y-coordinate of the segment's end point.

    Raises
    ------
    ValueError
        If the cutoff is located outside the parabola.

    Arguments
    ---------
    position : float
        Coordinate of the ellipse apex.
    radius : float
        Radius of curvature at the apex.
    cutoff : float
        x-coordinate at which the ellipse is cut off.
    return_endpoint : bool
        If true, returns a the y coordinate of the arc endpoint.
    """
    a = radius / 2  # The radius of curvature of a parabola is twice its focal length

    if (cutoff < position and radius > 0) or (cutoff > position and radius < 0):
        message = "The cutoff coordinate is located outside the domain of the parabola."
        raise ValueError(message)

    t_max = np.abs(np.sqrt((cutoff - position) / a))
    t = np.linspace(-t_max, t_max, 1000)

    vertices = list(zip(position + a * t**2, 2 * a * t))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)

    return Path(vertices, codes)


def _get_hyperbola_sizes(radius: float, conic: float) -> tuple[float, float]:
    return -radius / (conic + 1), radius / np.sqrt(-conic - 1)


def plot_hyperbola(position: float, radius: float, conic: float, cutoff: float) -> Path:
    """Plot a segment of a hyperbola.

    Creates a `Path` for a hyperbola. The radius of curvature at the apex is specified as `radius`.
    The hyperbola is cut off at the x-coordinate specified by `cutoff`.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the parabola segment.
    float
        y-coordinate of the segment's end point.

    Raises
    ------
    ValueError
        If the cutoff is located outside the parabola.

    Arguments
    ---------
    position : float
        Coordinate of the hyperbola apex.
    radius : float
        Radius of curvature at the apex.
    conic : float
        Conic constant (asphericity) of the hyperbola. Must be < -1.
    cutoff
        x-coordinate at which the hyperbola is cut off.
    """
    a, b = _get_hyperbola_sizes(radius, conic)

    # Correct position for hyperbola apex
    position -= a

    if (cutoff < position and radius > 0) or (cutoff > position and radius < 0):
        message = "The cutoff coordinate is located outside the domain of the hyperbola."
        raise ValueError(message)

    t_max = np.arccosh((cutoff - position) / a)
    t = np.linspace(-t_max, t_max, 1000)

    vertices = list(zip(position + a * np.cosh(t), b * np.sinh(t)))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)

    return Path(vertices, codes)


def _ellipse(x, rx, ry) -> float:
    """Upper segment of an ellipse. Used to find intersections between lens surfaces."""

    # Make sure the upper half is calculated
    ry = abs(ry)

    return ry * np.sqrt(1 - x**2 / rx**2)


def _parabola(x, rx) -> float:
    """Upper segment of a parabola. Used to find intersections between lens surfaces."""

    return 2 * np.sqrt(rx * x)


def _hyperbola(x, rx, ry) -> float:
    """Upper segment of a hyperbola. Used to find intersections between lens surfaces."""

    # Make sure the upper half is calculated
    ry = abs(ry)

    return ry * np.sqrt(x**2 / rx**2 - 1)


def _lens_surface_function(radius, conic, position) -> Callable[[float], float]:
    """Function for the upper segment of a lens surface. Used to find intersections between lens surfaces."""
    if conic < -1:  # Hyperbola
        rx, ry = _get_hyperbola_sizes(radius, conic)
        x0 = position - rx

        def surface_function(x: float) -> float:
            return _hyperbola(x - x0, rx, ry)

    elif conic == -1:  # Parabola
        rx = radius / 2
        x0 = position

        def surface_function(x: float) -> float:
            return _parabola(x - x0, rx)

    else:  # Ellipse (conic > -1)
        rx, ry = _get_ellipse_sizes(radius, conic)
        x0 = position + rx

        def surface_function(x: float) -> float:
            return _ellipse(x - x0, rx, ry)

    return surface_function


def plot_eye(
    ax: Axes,
    geometry: EyeModel | EyeGeometry,
    lens_edge_thickness: float = 0.0,
    retina_cutoff_position: float | None = None,
    **kwargs,
) -> Axes:
    """Plot an eye.

    Plot an eye with geometric parameters specified by an `EyeGeometry` object.
    The eye is oriented along the horizontal axis, with the pupil center located at `(0, 0)`.
    Additional translations and rotations can be applied using matplotlib patch transforms.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Matplotlib axes on which the eye will be drawn.
    geometry : EyeGeometry
        Specification of the eye's geometrical parameters.
    lens_edge_thickness : float
        Thickness of the lens at its edge, defaults to 0. If specified, the lens will be cut off at the point where
        its thickness equals this value.
    retina_cutoff_position : float
        Location to which the retina should be drawn. Defaults to the lens's posterior apex.

    Returns
    -------
    matplotlib.pyplot.Axes
        Modified axes with the eye plot

    Raises
    ------
    ValueError
        If `lens_edge_thickness` is less than 0 or `retina_cutoff_position` is located behind the retina.
    """
    # Extract geometry when an EyeModel is supplied
    geometry = geometry if isinstance(geometry, EyeGeometry) else geometry.geometry

    # Input data validation
    if lens_edge_thickness < 0:
        message = f"lens_edge_thickness should be a positive number, got {lens_edge_thickness}."
        raise ValueError(message)
    if (
        retina_cutoff_position is not None
        and retina_cutoff_position > geometry.lens_thickness + geometry.vitreous_thickness
    ):
        message = "retina_cutoff_position is located behind the retina."
        raise ValueError(message)

    # Cornea front
    cornea_front, cornea_cutoff_y = plot_surface(
        -1 * (geometry.cornea_thickness + geometry.anterior_chamber_depth),
        geometry.cornea_front.radius,
        geometry.cornea_front.asphericity,
        cutoff=0,
        return_endpoint=True,
    )

    # Cornea back
    cornea_back = plot_surface(
        -geometry.anterior_chamber_depth,
        geometry.cornea_back.radius,
        geometry.cornea_back.asphericity,
        cutoff=0,
    )

    # Retina
    retina = plot_surface(
        geometry.lens_thickness + geometry.vitreous_thickness,
        geometry.retina.radius,
        geometry.retina.asphericity,
        cutoff=geometry.lens_thickness if retina_cutoff_position is None else retina_cutoff_position,
    )

    # Solve with the lens front surface shifted forward by lens_edge_thickness
    lens_front_function = _lens_surface_function(
        geometry.lens_front.radius,
        geometry.lens_front.asphericity,
        0 + lens_edge_thickness,
    )
    lens_back_function = _lens_surface_function(
        geometry.lens_back.radius,
        geometry.lens_back.asphericity,
        geometry.lens_thickness,
    )

    lens_intersection = fsolve(
        lambda x: lens_front_function(x) - lens_back_function(x),
        x0=geometry.lens_thickness / 2,
    )[0]

    # Lens front
    lens_front = plot_surface(
        0,
        geometry.lens_front.radius,
        geometry.lens_front.asphericity,
        cutoff=lens_intersection - lens_edge_thickness,
    )

    # Lens back
    lens_back = plot_surface(
        geometry.lens_thickness,
        geometry.lens_back.radius,
        geometry.lens_back.asphericity,
        cutoff=lens_intersection,
    )

    # Lens edges
    if lens_edge_thickness > 0:
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = [
            lens_front.vertices[0],
            lens_front.vertices[0] + (lens_edge_thickness, 0),
            lens_front.vertices[-1],
            lens_front.vertices[-1] + (lens_edge_thickness, 0),
        ]
        lens_edges = Path(vertices, codes)
    else:
        lens_edges = Path(np.zeros((0, 2)))

    # Iris
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.MOVETO,
        Path.LINETO,
    ]
    vertices = [
        (0, cornea_cutoff_y),
        (0, geometry.pupil.semi_diameter),
        (0, -geometry.pupil.semi_diameter),
        (0, -cornea_cutoff_y),
    ]
    iris = Path(vertices, codes)

    eye = Path.make_compound_path(cornea_front, cornea_back, iris, lens_front, lens_back, lens_edges, retina)
    ax.add_patch(patches.PathPatch(eye, fill=None, **kwargs))

    return ax
