"""Bennett-Rabbetts schematic eye."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.models.geometry import EyeGeometry, EyeModelSurfaces, NoSurface, StandardSurface, Stop

if TYPE_CHECKING:
    from visisipy.types import Unpack

__all__ = ("BennettRabbettsGeometry",)


PUPIL_SEMI_DIAMETER_MM = 1.5


class BennettRabbettsSurfaces(EyeModelSurfaces, total=False):
    """Surfaces of the Bennett-Rabbetts schematic eye.

    This schematic eye does not have an anterior corneal surface.
    """

    cornea_front: NoSurface


surfaces_by_accommodation: dict[float, BennettRabbettsSurfaces] = {
    0: BennettRabbettsSurfaces(
        cornea_front=NoSurface(),
        cornea_back=StandardSurface(radius=7.80, asphericity=0, thickness=3.60),
        pupil=Stop(semi_diameter=PUPIL_SEMI_DIAMETER_MM),
        lens_front=StandardSurface(radius=11.0, asphericity=0, thickness=3.70),
        lens_back=StandardSurface(radius=-6.47515, asphericity=0, thickness=16.79),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    ),
    2.5: BennettRabbettsSurfaces(
        cornea_front=NoSurface(),
        cornea_back=StandardSurface(radius=7.80, asphericity=0, thickness=3.475),
        pupil=Stop(semi_diameter=PUPIL_SEMI_DIAMETER_MM),
        lens_front=StandardSurface(radius=8.60, asphericity=0, thickness=3.825),
        lens_back=StandardSurface(radius=-5.909, asphericity=0, thickness=16.79),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    ),
    5: BennettRabbettsSurfaces(
        cornea_front=NoSurface(),
        cornea_back=StandardSurface(radius=7.80, asphericity=0, thickness=3.37),
        pupil=Stop(semi_diameter=PUPIL_SEMI_DIAMETER_MM),
        lens_front=StandardSurface(radius=7.00, asphericity=0, thickness=3.93),
        lens_back=StandardSurface(radius=-5.504, asphericity=0, thickness=16.79),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    ),
    7.5: BennettRabbettsSurfaces(
        cornea_front=NoSurface(),
        cornea_back=StandardSurface(radius=7.80, asphericity=0, thickness=3.28),
        pupil=Stop(semi_diameter=PUPIL_SEMI_DIAMETER_MM),
        lens_front=StandardSurface(radius=6.00, asphericity=0, thickness=4.02),
        lens_back=StandardSurface(radius=-5.063, asphericity=0, thickness=16.79),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    ),
    10.0: BennettRabbettsSurfaces(
        cornea_front=NoSurface(),
        cornea_back=StandardSurface(radius=7.80, asphericity=0, thickness=3.21),
        pupil=Stop(semi_diameter=PUPIL_SEMI_DIAMETER_MM),
        lens_front=StandardSurface(radius=5.20, asphericity=0, thickness=4.09),
        lens_back=StandardSurface(radius=-4.750, asphericity=0, thickness=16.79),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    ),
}


class BennettRabbettsGeometry(
    EyeGeometry[NoSurface, StandardSurface, Stop, StandardSurface, StandardSurface, StandardSurface]
):
    """Geometric parameters of the Bennett-Rabbetts schematic eye.

    This schematic eye is based on the Bennett-Rabbetts model as described in [1]_.
    Sizes are specified in mm.

    Notes
    -----
    The Bennett-Rabbetts eye is a three-surface schematic eye with a single cornea
    surface. In visisipy, this means that only the posterior corneal surface is modelled,
    and the corneal thickness will be equal to zero.
    The Bennett-Rabbetts geometry does not specify the retinal curvature, so the value
    from the Navarro model is used by default.

    References
    ----------
    .. [1] Bennett, A. G., & Rabbetts, R. B. (1990). Clinical visual optics (2nd ed.). Butterworth-Heinemann.
    """

    def __init__(self, accommodation: float = 0.0, **surfaces: Unpack[BennettRabbettsSurfaces]) -> None:
        """Create a Bennett-Rabbetts schematic eye geometry.

        Parameters
        ----------
        accommodation : float
            Accommodation in diopters. Available values are 0, 2.5, 5, 7.5, and 10 D.
        """
        if accommodation not in surfaces_by_accommodation:
            msg = f"Accommodation value {accommodation} not available. Available values are: {list(surfaces_by_accommodation.keys())}."
            raise ValueError(msg)
        bennett_surfaces = surfaces_by_accommodation[accommodation].copy()
        bennett_surfaces.update(surfaces)

        super().__init__(**bennett_surfaces)
