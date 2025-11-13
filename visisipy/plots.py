"""Visualization tools for Visisipy."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from matplotlib import patches
from matplotlib.path import Path
from scipy.optimize import fsolve

from visisipy.models import EyeGeometry, EyeModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes


__all__ = ("plot_eye",)


@overload
def plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
    max_thickness: float = 15.0,
) -> Path: ...


@overload
def plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
    max_thickness: float = 15.0,
) -> tuple[Path, float]: ...


def plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
    max_thickness: float = 15.0,
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
    max_thickness : float
        Maximum thickness to draw when cutoff is unreachable (for robustness).
    """
    # Special case: radius = 0 means a flat (vertical) surface
    if radius == 0:
        # Draw a straight vertical line from position to cutoff
        vertices = [(position, -max_thickness), (position, max_thickness)]
        codes = [Path.MOVETO, Path.LINETO]
        flat_surface = Path(vertices, codes)
        return (flat_surface, max_thickness) if return_endpoint else flat_surface

    if conic > -1:
        return plot_ellipse(
            position, radius, conic, cutoff, return_endpoint=return_endpoint, max_thickness=max_thickness
        )
    if conic == -1:
        return plot_parabola(position, radius, cutoff, return_endpoint=return_endpoint, max_thickness=max_thickness)

    return plot_hyperbola(position, radius, conic, cutoff, return_endpoint=return_endpoint, max_thickness=max_thickness)


def _get_ellipse_sizes(radius: float, conic: float) -> tuple[float, float]:
    """Calculate rx and ry (axial and radial radii) of an ellipse from its radius of curvature and conic."""
    # Prolate or oblate does not matter in this case
    return radius / (conic + 1), radius / np.sqrt(conic + 1)


@overload
def plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
    max_thickness: float = 15.0,
) -> Path: ...


@overload
def plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
    max_thickness: float = 15.0,
) -> tuple[Path, float]: ...


def plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
    max_thickness: float = 15.0,
) -> Path | tuple[Path, float]:
    """Plot a segment of an ellipse.

    Creates an `Arc` patch for an ellipse. The radius of curvature at the apex is specified as `radius`.
    The ellipse is cut off at the x-coordinate specified by `cutoff`. If cutoff is unreachable,
    draws to max_thickness.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the ellipse segment.
    float
        y-coordinate of the segment's end point.

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
    max_thickness : float
        Maximum thickness to draw when cutoff is unreachable.
    """
    rx, ry = _get_ellipse_sizes(radius, conic)

    x0 = position + rx

    # Check if cutoff is reachable
    cutoff_relative = cutoff - x0
    if abs(cutoff_relative / rx) > 1:
        # Cutoff is beyond ellipse extent - clamp to maximum extent or use max_thickness
        if radius > 0:
            # Forward curving - limit by max_thickness
            max_x_from_thickness = position + max_thickness if abs(ry) >= max_thickness else position + 2 * abs(rx)
            cutoff = min(cutoff, max_x_from_thickness)
        else:
            # Backward curving - cutoff can't be beyond position (apex)
            cutoff = min(cutoff, position)

        # Recalculate relative position
        cutoff_relative = cutoff - x0
        if abs(cutoff_relative / rx) > 1:
            # Still out of range, clamp to valid domain
            cutoff_relative = np.sign(cutoff_relative) * abs(rx)

    t_max = np.abs(np.arccos(cutoff_relative / rx))
    # This is a bit weird, but necessary to draw the arc in the right direction
    t = np.linspace(-t_max, t_max - 2 * np.pi, 1000)

    vertices = list(zip(x0 + rx * np.cos(t), ry * np.sin(t), strict=False))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    ellipse = Path(vertices, codes)

    return (ellipse, ry * np.sin(t_max)) if return_endpoint else ellipse


@overload
def plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
    max_thickness: float = 15.0,
) -> Path: ...


@overload
def plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
    max_thickness: float = 15.0,
) -> tuple[Path, float]: ...


def plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
    max_thickness: float = 15.0,
) -> Path | tuple[Path, float]:
    """Plot a segment of a parabola.

    Creates a `Path` for a parabola. The radius of curvature at the apex is specified as `radius`.
    The parabola is cut off at the x-coordinate specified by `cutoff`. If cutoff is unreachable,
    draws to max_thickness.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the parabola segment.
    float
        y-coordinate of the segment's end point.

    Arguments
    ---------
    position : float
        Coordinate of the parabola apex.
    radius : float
        Radius of curvature at the apex.
    cutoff : float
        x-coordinate at which the parabola is cut off.
    return_endpoint : bool
        If true, returns the y coordinate of the arc endpoint.
    max_thickness : float
        Maximum thickness to draw when cutoff is unreachable.
    """
    a = radius / 2  # The radius of curvature of a parabola is twice its focal length

    if (cutoff < position and radius > 0) or (cutoff > position and radius < 0):
        # Cutoff is outside domain - use max_thickness constraint
        if radius > 0:
            cutoff = position + max_thickness**2 / (4 * abs(a))
        else:
            cutoff = position - max_thickness**2 / (4 * abs(a))

    t_max = np.abs(np.sqrt((cutoff - position) / a))
    t = np.linspace(-t_max, t_max, 1000)

    vertices = list(zip(position + a * t**2, 2 * a * t, strict=False))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    parabola = Path(vertices, codes)

    if return_endpoint:
        endpoint_y = 2 * a * t_max
        return parabola, endpoint_y

    return parabola


def _get_hyperbola_sizes(radius: float, conic: float) -> tuple[float, float]:
    return -radius / (conic + 1), radius / np.sqrt(-conic - 1)


@overload
def plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
    max_thickness: float = 15.0,
) -> Path: ...


@overload
def plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
    max_thickness: float = 15.0,
) -> tuple[Path, float]: ...


def plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
    max_thickness: float = 15.0,
) -> Path | tuple[Path, float]:
    """Plot a segment of a hyperbola.

    Creates a `Path` for a hyperbola. The radius of curvature at the apex is specified as `radius`.
    The hyperbola is cut off at the x-coordinate specified by `cutoff`. If cutoff is unreachable,
    draws to max_thickness.

    Returns
    -------
    Path
        A `matplotlib.path.Path` object with the hyperbola segment.
    float
        y-coordinate of the segment's end point.

    Arguments
    ---------
    position : float
        Coordinate of the hyperbola apex.
    radius : float
        Radius of curvature at the apex.
    conic : float
        Conic constant (asphericity) of the hyperbola. Must be < -1.
    cutoff : float
        x-coordinate at which the hyperbola is cut off.
    return_endpoint : bool
        If true, returns the y coordinate of the arc endpoint.
    max_thickness : float
        Maximum thickness to draw when cutoff is unreachable.
    """
    a, b = _get_hyperbola_sizes(radius, conic)

    # Correct position for hyperbola apex
    position -= a

    if (cutoff < position and radius > 0) or (cutoff > position and radius < 0):
        # Cutoff is outside domain - use max_thickness constraint
        if radius > 0:
            cutoff = position + abs(a) * np.sqrt(1 + (max_thickness / abs(b)) ** 2)
        else:
            cutoff = position - abs(a) * np.sqrt(1 + (max_thickness / abs(b)) ** 2)

    t_max = np.arccosh((cutoff - position) / a)
    t = np.linspace(-t_max, t_max, 1000)

    vertices = list(zip(position + a * np.cosh(t), b * np.sinh(t), strict=False))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    hyperbola = Path(vertices, codes)

    if return_endpoint:
        endpoint_y = b * np.sinh(t_max)
        return hyperbola, endpoint_y

    return hyperbola


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


def _is_convex(radius: float) -> bool:
    """Determine if a surface is convex (positive radius)."""
    return radius > 0


def _is_concave(radius: float) -> bool:
    """Determine if a surface is concave (negative radius)."""
    return radius < 0


def _find_intersection(func1: Callable[[float], float], func2: Callable[[float], float], x0: float) -> float | None:
    """Find intersection of two surface functions. Returns None if no intersection found."""
    try:
        result = fsolve(lambda x: func1(x) - func2(x), x0=x0, full_output=True)
        x_intersect, info, ier, _ = result
        if ier == 1:  # Solution found
            return float(x_intersect[0])
    except (ValueError, RuntimeWarning):
        pass
    return None


def _match_surface_vertices(front_surface: Path, back_surface: Path) -> list:
    """Match vertices from two surfaces by y-coordinate to ensure proper edge connections.

    Returns list of vertices for connecting edges: [front_top, back_top, front_bottom, back_bottom]
    or [front_top, back_bottom, front_bottom, back_top] depending on which pairing minimizes distance.
    """
    # Get y-coordinates of endpoints
    front_y0 = front_surface.vertices[0][1]
    _ = front_surface.vertices[-1][1]
    back_y0 = back_surface.vertices[0][1]
    back_y1 = back_surface.vertices[-1][1]

    # Match vertices: if front_y0 and back_y0 are closer, connect 0-0 and -1--1
    # Otherwise connect 0--1 and -1-0
    if abs(front_y0 - back_y0) < abs(front_y0 - back_y1):
        # Connect matching ends
        return [
            front_surface.vertices[0],
            back_surface.vertices[0],
            front_surface.vertices[-1],
            back_surface.vertices[-1],
        ]
    # Connect opposite ends
    return [
        front_surface.vertices[0],
        back_surface.vertices[-1],
        front_surface.vertices[-1],
        back_surface.vertices[0],
    ]


def _get_max_radius_cutoff(position: float, radius: float, conic: float) -> float:
    """Calculate the x-position where a conic surface reaches its maximum vertical radius.

    For an ellipse with negative radius (concave), this is where the semi-minor axis is located.
    """
    if conic > -1:  # Ellipse
        rx, _ = _get_ellipse_sizes(radius, conic)
        # For a concave surface (negative radius), the maximum radius is at position + rx
        # For a convex surface (positive radius), the maximum radius is at position + rx
        return position + rx
    if conic == -1:  # Parabola
        # For parabola, use a reasonable default based on max_thickness
        a = radius / 2
        max_thickness = 15.0
        if radius > 0:
            return position + max_thickness**2 / (4 * abs(a))
        return position - max_thickness**2 / (4 * abs(a))
    # Hyperbola
    a, b = _get_hyperbola_sizes(radius, conic)
    max_thickness = 15.0
    # Adjust position for hyperbola apex
    position -= a
    if radius > 0:
        return position + abs(a) * np.sqrt(1 + (max_thickness / abs(b)) ** 2)
    return position - abs(a) * np.sqrt(1 + (max_thickness / abs(b)) ** 2)


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
        Location to which the retina should be drawn. Defaults to the lens's posterior apex. For a value of `0`, the
        cutoff is located at the pupil.

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

    # Positions
    cornea_front_pos = -1 * (geometry.cornea_thickness + geometry.anterior_chamber_depth)
    cornea_back_pos = -geometry.anterior_chamber_depth
    pupil_pos = 0.0
    lens_front_pos = pupil_pos + geometry.pupil_lens_distance
    lens_back_pos = lens_front_pos + geometry.lens_thickness
    retina_pos = lens_back_pos + geometry.vitreous_thickness

    # Determine curvatures
    cornea_front_convex = _is_convex(geometry.cornea_front.radius)
    cornea_back_concave = _is_concave(geometry.cornea_back.radius)
    lens_front_convex = _is_convex(geometry.lens_front.radius)
    lens_back_concave = _is_concave(geometry.lens_back.radius)
    retina_concave = _is_concave(geometry.retina.radius)

    # CORNEA LOGIC
    cornea_edges = Path(np.zeros((0, 2)))

    if cornea_front_convex and cornea_back_concave:
        # Normal convex-concave cornea: both cut at pupil, connect with lines
        cornea_front, cornea_front_y = plot_surface(
            cornea_front_pos,
            geometry.cornea_front.radius,
            geometry.cornea_front.asphericity,
            cutoff=pupil_pos,
            return_endpoint=True,
        )
        cornea_back, cornea_back_y = plot_surface(
            cornea_back_pos,
            geometry.cornea_back.radius,
            geometry.cornea_back.asphericity,
            cutoff=pupil_pos,
            return_endpoint=True,
        )
        # Connect cornea surfaces
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = _match_surface_vertices(cornea_front, cornea_back)
        cornea_edges = Path(vertices, codes)
        cornea_cutoff_y = cornea_front_y

    elif cornea_front_convex and not cornea_back_concave:
        # Biconvex cornea: cut at intersection
        cf_func = _lens_surface_function(
            geometry.cornea_front.radius,
            geometry.cornea_front.asphericity,
            cornea_front_pos,
        )
        cb_func = _lens_surface_function(
            geometry.cornea_back.radius,
            geometry.cornea_back.asphericity,
            cornea_back_pos,
        )
        cornea_intersection = _find_intersection(cf_func, cb_func, (cornea_front_pos + cornea_back_pos) / 2)

        if cornea_intersection is not None:
            cornea_front, cornea_front_y = plot_surface(
                cornea_front_pos,
                geometry.cornea_front.radius,
                geometry.cornea_front.asphericity,
                cutoff=cornea_intersection,
                return_endpoint=True,
            )
            cornea_back, cornea_back_y = plot_surface(
                cornea_back_pos,
                geometry.cornea_back.radius,
                geometry.cornea_back.asphericity,
                cutoff=cornea_intersection,
                return_endpoint=True,
            )
            cornea_cutoff_y = cornea_front_y
        else:
            # Fallback to pupil cutoff
            cornea_front, cornea_front_y = plot_surface(
                cornea_front_pos,
                geometry.cornea_front.radius,
                geometry.cornea_front.asphericity,
                cutoff=pupil_pos,
                return_endpoint=True,
            )
            cornea_back, cornea_back_y = plot_surface(
                cornea_back_pos,
                geometry.cornea_back.radius,
                geometry.cornea_back.asphericity,
                cutoff=pupil_pos,
                return_endpoint=True,
            )
            codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
            vertices = _match_surface_vertices(cornea_front, cornea_back)
            cornea_edges = Path(vertices, codes)
            cornea_cutoff_y = cornea_front_y

    else:
        # Unusual configuration (concave front)
        warnings.warn("Concave cornea front detected. Drawing to maximum thickness of 5 mm.", stacklevel=2)
        # Concave cornea curves backward, so extend 5mm backward from apex
        cornea_front, cornea_front_y = plot_surface(
            cornea_front_pos,
            geometry.cornea_front.radius,
            geometry.cornea_front.asphericity,
            cutoff=cornea_front_pos - 5.0,
            return_endpoint=True,
            max_thickness=5.0,
        )
        cornea_back, cornea_back_y = plot_surface(
            cornea_back_pos,
            geometry.cornea_back.radius,
            geometry.cornea_back.asphericity,
            cutoff=pupil_pos,
            return_endpoint=True,
        )
        # Connect surfaces
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = _match_surface_vertices(cornea_front, cornea_back)
        cornea_edges = Path(vertices, codes)
        cornea_cutoff_y = max(abs(cornea_front_y), abs(cornea_back_y))

    # LENS LOGIC
    lens_edges = Path(np.zeros((0, 2)))

    if lens_front_convex and lens_back_concave:
        # Normal biconvex lens: cut at intersection (or edge thickness point)
        lf_func = _lens_surface_function(
            geometry.lens_front.radius,
            geometry.lens_front.asphericity,
            lens_front_pos + lens_edge_thickness,
        )
        lb_func = _lens_surface_function(
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            lens_back_pos,
        )
        lens_intersection = _find_intersection(lf_func, lb_func, (lens_front_pos + lens_back_pos) / 2)

        if lens_intersection is not None:
            lens_front = plot_surface(
                lens_front_pos,
                geometry.lens_front.radius,
                geometry.lens_front.asphericity,
                cutoff=lens_intersection - lens_edge_thickness,
            )
            lens_back = plot_surface(
                lens_back_pos,
                geometry.lens_back.radius,
                geometry.lens_back.asphericity,
                cutoff=lens_intersection,
            )

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
            # No intersection - draw to apexes and connect
            lens_front = plot_surface(
                lens_front_pos,
                geometry.lens_front.radius,
                geometry.lens_front.asphericity,
                cutoff=lens_back_pos,
            )
            lens_back = plot_surface(
                lens_back_pos,
                geometry.lens_back.radius,
                geometry.lens_back.asphericity,
                cutoff=lens_back_pos,
            )
            codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
            vertices = [
                lens_front.vertices[0],
                lens_back.vertices[0],
                lens_front.vertices[-1],
                lens_back.vertices[-1],
            ]
            lens_edges = Path(vertices, codes)

    elif not lens_front_convex and not lens_back_concave:
        # Biconcave lens: front at pupil, back at retina intersection
        lb_func = _lens_surface_function(
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            lens_back_pos,
        )
        r_func = _lens_surface_function(
            geometry.retina.radius,
            geometry.retina.asphericity,
            retina_pos,
        )
        retina_intersection = _find_intersection(lb_func, r_func, (lens_back_pos + retina_pos) / 2)

        lens_front = plot_surface(
            lens_front_pos,
            geometry.lens_front.radius,
            geometry.lens_front.asphericity,
            cutoff=pupil_pos,
        )
        lens_back = plot_surface(
            lens_back_pos,
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            cutoff=retina_intersection if retina_intersection is not None else retina_pos,
        )
        # Connect surfaces - match by y-coordinate (top to top, bottom to bottom)
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = _match_surface_vertices(lens_front, lens_back)
        lens_edges = Path(vertices, codes)

    elif lens_front_convex and not lens_back_concave:
        # Convex-concave lens: like cornea but cut at 2x lens thickness
        max_cutoff = lens_front_pos + 2 * geometry.lens_thickness
        lens_front = plot_surface(
            lens_front_pos,
            geometry.lens_front.radius,
            geometry.lens_front.asphericity,
            cutoff=max_cutoff,
        )
        lens_back = plot_surface(
            lens_back_pos,
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            cutoff=max_cutoff,
        )
        # Connect surfaces
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = _match_surface_vertices(lens_front, lens_back)
        lens_edges = Path(vertices, codes)

    else:
        # Concave-convex lens: both at pupil
        lens_front = plot_surface(
            lens_front_pos,
            geometry.lens_front.radius,
            geometry.lens_front.asphericity,
            cutoff=pupil_pos,
        )
        lens_back = plot_surface(
            lens_back_pos,
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            cutoff=pupil_pos,
        )
        # Connect surfaces
        codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
        vertices = _match_surface_vertices(lens_front, lens_back)
        lens_edges = Path(vertices, codes)

    # RETINA LOGIC
    if retina_concave and lens_back_concave:
        # Normal: concave retina, convex lens back - cut at lens back apex
        retina_cutoff = lens_back_pos if retina_cutoff_position is None else retina_cutoff_position
        retina = plot_surface(
            retina_pos,
            geometry.retina.radius,
            geometry.retina.asphericity,
            cutoff=retina_cutoff,
        )
    elif retina_concave and not lens_back_concave:
        # Concave retina, concave lens back - cut at intersection
        lb_func = _lens_surface_function(
            geometry.lens_back.radius,
            geometry.lens_back.asphericity,
            lens_back_pos,
        )
        r_func = _lens_surface_function(
            geometry.retina.radius,
            geometry.retina.asphericity,
            retina_pos,
        )
        retina_intersection = _find_intersection(lb_func, r_func, (lens_back_pos + retina_pos) / 2)
        if retina_intersection is not None:
            retina_cutoff = retina_intersection
        else:
            # No intersection - cut at maximum vertical radius to avoid complete circle
            retina_cutoff = _get_max_radius_cutoff(retina_pos, geometry.retina.radius, geometry.retina.asphericity)
        if retina_cutoff_position is not None:
            retina_cutoff = retina_cutoff_position
        retina = plot_surface(
            retina_pos,
            geometry.retina.radius,
            geometry.retina.asphericity,
            cutoff=retina_cutoff,
        )
    else:
        # Convex retina - draw to 5mm thickness and warn
        warnings.warn("Convex retina detected. Drawing to maximum thickness of 5 mm.", stacklevel=2)
        # Convex retina curves forward, so extend 5mm forward from apex
        retina_cutoff = retina_pos + 5.0 if retina_cutoff_position is None else retina_cutoff_position
        retina = plot_surface(
            retina_pos,
            geometry.retina.radius,
            geometry.retina.asphericity,
            cutoff=retina_cutoff,
            max_thickness=5.0,
        )

    # IRIS (connects cornea surfaces at pupil)
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

    eye = Path.make_compound_path(
        cornea_front, cornea_back, cornea_edges, iris, lens_front, lens_back, lens_edges, retina
    )
    ax.add_patch(patches.PathPatch(eye, fill=None, **kwargs))

    return ax
