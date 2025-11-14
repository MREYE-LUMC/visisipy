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

    from visisipy.models.geometry import StandardSurface


__all__ = ("plot_eye",)


@overload
def _plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
) -> Path: ...


@overload
def _plot_surface(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
) -> tuple[Path, float]: ...


def _plot_surface(
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
    max_thickness : float
        Maximum thickness to draw when cutoff is unreachable.
    """
    # Special case: radius = 0 means a flat (vertical) surface
    if radius == 0:
        # Draw a straight vertical line from position to cutoff
        vertices = [(position, -np.inf), (position, np.inf)]
        codes = [Path.MOVETO, Path.LINETO]
        flat_surface = Path(vertices, codes)
        return (flat_surface, np.inf) if return_endpoint else flat_surface

    if conic > -1:
        return _plot_ellipse(position, radius, conic, cutoff, return_endpoint=return_endpoint)
    if conic == -1:
        return _plot_parabola(position, radius, cutoff, return_endpoint=return_endpoint)

    return _plot_hyperbola(position, radius, conic, cutoff, return_endpoint=return_endpoint)


def _get_ellipse_sizes(radius: float, conic: float) -> tuple[float, float]:
    """Calculate rx and ry (axial and radial radii) of an ellipse from its radius of curvature and conic."""
    # Prolate or oblate does not matter in this case
    return radius / (conic + 1), radius / np.sqrt(conic + 1)


@overload
def _plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
) -> Path: ...


@overload
def _plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
) -> tuple[Path, float]: ...


def _plot_ellipse(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
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
    cutoff_relative = cutoff - x0

    if abs(cutoff_relative / rx) > 1:
        message = f"Cutoff is located outside the ellipse: {cutoff=}, {rx=}"
        raise ValueError(message)

    t_max = np.abs(np.arccos(cutoff_relative / rx))
    # This is a bit weird, but necessary to draw the arc in the right direction
    t = np.linspace(-t_max, t_max - 2 * np.pi, 1000)

    vertices = list(zip(x0 + rx * np.cos(t), ry * np.sin(t), strict=False))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 1)
    ellipse = Path(vertices, codes)

    return (ellipse, ry * np.sin(t_max)) if return_endpoint else ellipse


@overload
def _plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
) -> Path: ...


@overload
def _plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
) -> tuple[Path, float]: ...


def _plot_parabola(
    position: float,
    radius: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
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
        message = "Cutoff is outside the domain of the parabola."
        raise ValueError(message)

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
def _plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[False] = False,
) -> Path: ...


@overload
def _plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: Literal[True],
) -> tuple[Path, float]: ...


def _plot_hyperbola(
    position: float,
    radius: float,
    conic: float,
    cutoff: float,
    *,
    return_endpoint: bool = False,
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
        message = "The cutoff coordinate is located outside the domain of the hyperbola."
        raise ValueError(message)

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


def _is_ellipse(surface: StandardSurface) -> bool:
    """Determine if a surface is an ellipse (conic > -1)."""
    return surface.asphericity > -1


def _is_convex(radius: float, side: Literal["image", "object"]) -> bool:
    """Determine if a surface is convex (positive radius)."""
    if side == "object":
        return radius > 0

    if side == "image":
        return radius < 0

    raise ValueError("side must be either 'image' or 'object'")


def _is_concave(radius: float, side: Literal["image", "object"]) -> bool:
    """Determine if a surface is concave (negative radius)."""
    if radius == 0:
        return False

    return not _is_convex(radius, side)


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


def _plot_cornea(
    geometry: EyeGeometry,
    cornea_front_pos: float,
    cornea_back_pos: float,
    pupil_pos: float,
) -> tuple[Path, float]:
    cornea_front_convex = _is_convex(geometry.cornea_front.radius, side="object")
    cornea_back_concave = _is_concave(geometry.cornea_back.radius, side="image")

    cornea_front_cutoff: float
    cornea_back_cutoff: float

    if cornea_front_convex and cornea_back_concave:
        # Normal convex-concave cornea: both cut at pupil, connect with lines
        cornea_front_cutoff = cornea_back_cutoff = pupil_pos
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
        cornea_front_cutoff = cornea_back_cutoff = pupil_pos if cornea_intersection is None else cornea_intersection
    else:
        # Unusual configuration (concave front)
        warnings.warn("Concave cornea front detected. Drawing to maximum thickness of 5 mm.", stacklevel=2)
        # Concave cornea curves backward, so extend 5mm backward from apex
        cornea_front_cutoff = cornea_front_pos - 5.0

        # For concave-convex cornea, back should be cut at same location as front
        # For other cases, cut at pupil
        cornea_back_cutoff = pupil_pos if cornea_back_concave else cornea_front_cutoff

    cornea_front, cornea_front_y = _plot_surface(
        cornea_front_pos,
        geometry.cornea_front.radius,
        geometry.cornea_front.asphericity,
        cutoff=cornea_front_cutoff,
        return_endpoint=True,
    )
    cornea_back, cornea_back_y = _plot_surface(
        cornea_back_pos,
        geometry.cornea_back.radius,
        geometry.cornea_back.asphericity,
        cutoff=cornea_back_cutoff,
        return_endpoint=True,
    )
    # Connect cornea surfaces
    edge_codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
    edge_vertices = _match_surface_vertices(cornea_front, cornea_back)
    cornea_edges = Path(edge_vertices, edge_codes)
    cornea_cutoff_y = cornea_front_y if cornea_front_convex else max(abs(cornea_front_y), abs(cornea_back_y))

    cornea = Path.make_compound_path(cornea_front, cornea_back, cornea_edges)

    return cornea, cornea_cutoff_y


def _plot_lens(
    geometry: EyeGeometry,
    pupil_pos: float,
    lens_front_pos: float,
    lens_back_pos: float,
    retina_pos: float,
    lens_edge_thickness: float,
) -> Path:
    lens_front_convex = _is_convex(geometry.lens_front.radius, side="object")
    lens_back_convex = _is_convex(geometry.lens_back.radius, side="image")

    lens_front_cutoff: float
    lens_back_cutoff: float

    if lens_front_convex and lens_back_convex:
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
            lens_front_cutoff = lens_intersection - lens_edge_thickness
            lens_back_cutoff = lens_intersection
        else:
            # No intersection - draw to apexes and connect
            lens_front_cutoff = lens_front_pos
            lens_back_cutoff = lens_back_pos
    elif not lens_front_convex and not lens_back_convex:
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

        lens_front_cutoff = pupil_pos
        lens_back_cutoff = retina_intersection if retina_intersection is not None else retina_pos
    elif lens_front_convex and not lens_back_convex:
        # Convex-concave lens: like cornea but cut at 2x lens thickness
        lens_front_cutoff = lens_back_cutoff = lens_front_pos + 2 * geometry.lens_thickness
    else:
        # Concave-convex lens: both at pupil
        lens_front_cutoff = lens_back_cutoff = pupil_pos

    lens_front = _plot_surface(
        lens_front_pos,
        geometry.lens_front.radius,
        geometry.lens_front.asphericity,
        cutoff=lens_front_cutoff,
    )
    lens_back = _plot_surface(
        lens_back_pos,
        geometry.lens_back.radius,
        geometry.lens_back.asphericity,
        cutoff=lens_back_cutoff,
    )

    edge_codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
    edge_vertices = _match_surface_vertices(lens_front, lens_back)
    lens_edges = Path(edge_vertices, edge_codes)

    return Path.make_compound_path(lens_front, lens_back, lens_edges)


def _plot_retina(
    geometry: EyeGeometry,
    lens_back_pos: float,
    retina_pos: float,
    retina_cutoff_position: float | None = None,
) -> Path:
    lens_back_convex = _is_convex(geometry.lens_back.radius, side="image")
    retina_concave = _is_concave(geometry.retina.radius, side="object")

    cutoff: float

    if retina_cutoff_position is not None:
        # Use the specified cutoff position
        cutoff = retina_cutoff_position

    elif retina_concave and lens_back_convex:
        # Normal: concave retina, convex lens back - cut at lens back apex
        if _is_ellipse(geometry.retina) and abs(retina_pos - lens_back_pos) > 2 * abs(
            geometry.retina.ellipsoid_radii.anterior_posterior
        ):
            # Distance between lens back and retina is more than retina diameter - cut at max radius
            # Retina is concave, so the radius is already negative
            cutoff = retina_pos + geometry.retina.ellipsoid_radii.anterior_posterior
        else:
            # Cut at lens back apex
            cutoff = lens_back_pos

    elif retina_concave and not lens_back_convex:
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
            cutoff = retina_intersection
        else:
            # No intersection - cut at maximum vertical radius to avoid complete circle
            # For concave retina, only draw posterior half (cutoff at or behind retina apex)
            max_cutoff = retina_pos - geometry.retina.ellipsoid_radii.anterior_posterior
            # For negative radius (concave), ensure we don't go beyond the apex (only posterior half)
            cutoff = min(max_cutoff, retina_pos) if geometry.retina.radius < 0 else max_cutoff
    else:
        # Convex retina - draw to 5mm thickness and warn
        warnings.warn("Convex retina detected. Drawing to maximum thickness of 5 mm.", stacklevel=2)
        # Convex retina curves forward, so extend 5mm forward from apex
        cutoff = retina_pos + 5.0 if retina_cutoff_position is None else retina_cutoff_position

    return _plot_surface(
        retina_pos,
        geometry.retina.radius,
        geometry.retina.asphericity,
        cutoff=cutoff,
    )


def _set_axis_limits(
    ax: Axes, cornea_front_position: float, retina_position: float, retina_radius: float, padding: float = 3.0
) -> None:
    """Set axis limits to fit the eye geometry with some padding.

    If the current limits are larger, they are preserved.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Matplotlib axes to set limits on.
    cornea_front_position : float
        Position of the anterior corneal surface.
    retina_position : float
        Position of the retina.
    retina_radius : float
        Transversal radius of the retina.
    padding : float
        Padding in mm to add around the eye geometry.
    """
    if padding < 0:
        raise ValueError("Padding must be positive.")

    current_xlims = ax.get_xlim()
    current_ylims = ax.get_ylim()

    x_min = cornea_front_position - padding
    x_max = retina_position + padding
    y_max = abs(retina_radius) + padding
    y_min = -y_max

    ax.set_xlim(min(current_xlims[0], x_min), max(current_xlims[1], x_max))
    ax.set_ylim(min(current_ylims[0], y_min), max(current_ylims[1], y_max))


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

    cornea, cornea_cutoff_y = _plot_cornea(
        geometry,
        cornea_front_pos,
        cornea_back_pos,
        pupil_pos,
    )
    lens = _plot_lens(
        geometry,
        pupil_pos,
        lens_front_pos,
        lens_back_pos,
        retina_pos,
        lens_edge_thickness,
    )
    retina = _plot_retina(
        geometry,
        lens_back_pos,
        retina_pos,
        retina_cutoff_position,
    )

    # Pupil
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
    pupil = Path(vertices, codes)

    eye = Path.make_compound_path(cornea, pupil, lens, retina)
    ax.add_patch(patches.PathPatch(eye, fill=None, **kwargs))

    _set_axis_limits(ax, cornea_front_pos, retina_pos, geometry.retina.ellipsoid_radii.inferior_superior)

    return ax
