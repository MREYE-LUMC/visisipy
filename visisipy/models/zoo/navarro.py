"""Escudero-Sanz and Navarro wide-angle schematic eye."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.models.geometry import EyeGeometry, EyeModelSurfaces, StandardSurface, Stop

if TYPE_CHECKING:
    from visisipy.types import Unpack


__all__ = ("NavarroGeometry",)


class NavarroGeometry(
    EyeGeometry[StandardSurface, StandardSurface, Stop, StandardSurface, StandardSurface, StandardSurface]
):
    """Geometric parameters of the Navarro wide-angle schematic eye.

    This schematic eye is based on the Navarro model as described in [1]_.
    Sizes are specified in mm.

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

    References
    ----------
    .. [1] Escudero-Sanz, I., & Navarro, R. (1999). Off-axis aberrations of a wide-angle schematic eye model.
        JOSA A, 16(8), 1881-1891. https://doi.org/10.1364/JOSAA.16.001881

    Examples
    --------
    Use the default Navarro geometry:

    >>> from visisipy import NavarroGeometry
    >>> geometry = NavarroGeometry()

    Create a Navarro geometry with a custom retina:

    >>> geometry = NavarroGeometry(
    ...     retina=StandardSurface(radius=-12.5, asphericity=0.5)
    ... )

    Create a default Navarro geometry and change only the lens back radius:

    >>> geometry = NavarroGeometry()
    >>> geometry.lens_back.radius = -5.8
    """

    def __init__(self, **surfaces: Unpack[EyeModelSurfaces]) -> None:
        navarro_surfaces = EyeModelSurfaces(
            cornea_front=StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55),
            cornea_back=StandardSurface(radius=6.50, asphericity=0, thickness=3.05),
            pupil=Stop(semi_diameter=1.348),
            lens_front=StandardSurface(radius=10.2, asphericity=-3.1316, thickness=4.0),
            lens_back=StandardSurface(radius=-6.0, asphericity=-1, thickness=16.3203),
            retina=StandardSurface(radius=-12.0, asphericity=0),
        )
        navarro_surfaces.update(**surfaces)
        super().__init__(**navarro_surfaces)
