"""Gullstrand-LeGrand schematic eye."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from visisipy.models.base import EyeModel
from visisipy.models.geometry import EyeGeometry, EyeModelSurfaces, StandardSurface, Stop
from visisipy.models.materials import GullstrandLeGrandAccommodatedMaterials, GullstrandLeGrandUnaccommodatedMaterials

if TYPE_CHECKING:
    from visisipy.types import Unpack

__all__ = ("GullstrandLeGrandEyeModel", "GullstrandLeGrandGeometry")

Accommodation = Literal["accommodated", "unaccommodated"]
surfaces_by_accommodation: dict[Accommodation, EyeModelSurfaces] = {
    "unaccommodated": EyeModelSurfaces(
        cornea_front=StandardSurface(radius=7.8, asphericity=0, thickness=0.55),
        cornea_back=StandardSurface(radius=6.5, asphericity=0, thickness=3.05),
        pupil=Stop(semi_diameter=1.5),
        lens_front=StandardSurface(radius=10.2, asphericity=0, thickness=4.0),
        lens_back=StandardSurface(radius=-6.0, asphericity=0, thickness=16.59655),
        retina=StandardSurface(radius=-13.4, asphericity=0),
    ),
    "accommodated": EyeModelSurfaces(
        cornea_front=StandardSurface(radius=7.8, asphericity=0, thickness=0.55),
        cornea_back=StandardSurface(radius=6.5, asphericity=0, thickness=2.65),
        pupil=Stop(semi_diameter=1.5),
        lens_front=StandardSurface(radius=6.0, asphericity=0, thickness=4.5),
        lens_back=StandardSurface(radius=-5.5, asphericity=0, thickness=16.49655),
        retina=StandardSurface(radius=-13.4, asphericity=0),
    ),
}


class GullstrandLeGrandGeometry(
    EyeGeometry[StandardSurface, StandardSurface, Stop, StandardSurface, StandardSurface, StandardSurface]
):
    """Geometric parameters of the Gullstrand-LeGrand schematic eye.

    This schematic eye is based on the Gullstrand-LeGrand model as described in [1]_.
    Sizes are specified in mm.

    References
    ----------
    .. [1] Le Grand, Y., El Hage, S.G. (1980). Physiological Optics. Springer.
    """

    def __init__(self, accommodation: Accommodation = "unaccommodated", **surfaces: Unpack[EyeModelSurfaces]) -> None:
        """Create a Gullstrand-LeGrand schematic eye geometry.

        Parameters
        ----------
        accommodation : {'accommodated', 'unaccommodated'}
            The accommodation state of the eye. Must be either 'accommodated' or 'unaccommodated'.

        Raises
        ------
        ValueError
            If `accommodation` is not 'accommodated' or 'unaccommodated'.
        """
        if accommodation not in surfaces_by_accommodation:
            msg = f"accommodation must be 'accommodated' or 'unaccommodated', got {accommodation}"
            raise ValueError(msg)

        gullstrand_surfaces = surfaces_by_accommodation[accommodation].copy()
        gullstrand_surfaces.update(**surfaces)

        super().__init__(**gullstrand_surfaces)


class GullstrandLeGrandEyeModel(EyeModel):
    """Gullstrand-LeGrand schematic eye model.

    See Also
    --------
    GullstrandLeGrandGeometry : Geometric parameters of the Gullstrand-LeGrand schematic eye.
    GullstrandLeGrandAccommodatedMaterials : Materials for the accommodated Gullstrand-LeGrand eye.
    GullstrandLeGrandUnaccommodatedMaterials : Materials for the unaccommodated Gullstrand-LeGrand eye.
    """

    def __init__(self, accommodation: Accommodation = "unaccommodated") -> None:
        """Create a Gullstrand-LeGrand schematic eye model.

        Parameters
        ----------
        accommodation : {'accommodated', 'unaccommodated'}
            The accommodation state of the eye. Must be either 'accommodated' or 'unaccommodated'.
        """
        geometry = GullstrandLeGrandGeometry(accommodation=accommodation)

        match accommodation:
            case "accommodated":
                materials = GullstrandLeGrandAccommodatedMaterials()
            case "unaccommodated":
                materials = GullstrandLeGrandUnaccommodatedMaterials()
            case _:
                msg = f"accommodation must be 'accommodated' or 'unaccommodated', got {accommodation}"
                raise ValueError(msg)

        super().__init__(geometry=geometry, materials=materials)
