"""Build eye models in Optiland."""

from __future__ import annotations

from typing import TYPE_CHECKING

from visisipy.models import BaseEye
from visisipy.optiland.surfaces import OptilandSurface, make_surface

if TYPE_CHECKING:
    from optiland.optic import Optic

    from visisipy import EyeModel


__all__ = ("OptilandEye",)


class OptilandEye(BaseEye):
    """Eye model in Optiland."""

    def __init__(self, eye_model: EyeModel) -> None:
        """Create a new Optiland eye model.

        Parameters
        ----------
        eye_model : EyeModel
            Eye model specification frin which the Optiland eye model is created.
        """
        self._eye_model = eye_model

        self._cornea_front = make_surface(eye_model.geometry.cornea_front, eye_model.materials.cornea, "cornea front")
        self._cornea_back = make_surface(
            eye_model.geometry.cornea_back, eye_model.materials.aqueous, "cornea back / aqueous"
        )
        self._pupil = make_surface(eye_model.geometry.pupil, eye_model.materials.aqueous, "pupil")
        self._lens_front = make_surface(eye_model.geometry.lens_front, eye_model.materials.lens, "lens front")
        self._lens_back = make_surface(
            eye_model.geometry.lens_back,
            eye_model.materials.vitreous,
            "lens back / vitreous",
        )
        self._retina = make_surface(eye_model.geometry.retina, eye_model.materials.vitreous, "retina")

    @property
    def eye_model(self) -> EyeModel:
        """Eye model specification from which the Optiland eye model is created."""
        return self._eye_model

    @property
    def cornea_front(self) -> OptilandSurface:
        """Cornea front surface."""
        return self._cornea_front

    @property
    def cornea_back(self) -> OptilandSurface:
        """Cornea back surface."""
        return self._cornea_back

    @property
    def pupil(self) -> OptilandSurface:
        """Pupil surface."""
        return self._pupil

    @property
    def lens_front(self) -> OptilandSurface:
        """Lens front surface."""
        return self._lens_front

    @property
    def lens_back(self) -> OptilandSurface:
        """Lens back surface."""
        return self._lens_back

    @property
    def retina(self) -> OptilandSurface:
        """Retina surface."""
        return self._retina

    def build(
        self,
        optic: Optic,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
    ) -> None:
        """Create the eye in Optiland.

        Create the eye model in the provided optical system `optic`, starting from `start_from_index`.
        If `replace_existing` is set to `True`, existing surfaces will be overwritten.

        Parameters
        ----------
        optic : Optic
            Optiland Optic in which the eye model is created.
        start_from_index : int
            Index of the surface after which the eye model will be built.
        replace_existing : bool
            If `True`, replaces existing surfaces instead of inserting new ones. Defaults to `False`.
        object_distance : float, optional
            Distance from the object surface (or the surface before the eye model) to the eye model. Defaults to infinity.

        Raises
        ------
        AssertionError
            If the pupil is not located at the stop position.
            If the retina is not located at the last surface.
        """
        # Create an object surface if it does not exist
        if optic.surface_group.num_surfaces != 0 and object_distance != float("inf"):
            raise ValueError("Cannot set a custom object distance if the optical system is not empty.")

        if start_from_index == 0 and optic.surface_group.num_surfaces == 0:
            optic.surface_group.add_surface(
                index=start_from_index, replace_existing=replace_existing, thickness=object_distance
            )
        elif start_from_index > optic.surface_group.num_surfaces - 1:
            message = "'start_from_index' can be at most the index of the last surface in the system."
            raise ValueError(message)

        cornea_front_index = self.cornea_front.build(
            optic, position=start_from_index + 1, replace_existing=replace_existing
        )
        cornea_back_index = self.cornea_back.build(
            optic, position=cornea_front_index + 1, replace_existing=replace_existing
        )
        pupil_index = cornea_back_index + 1
        pupil_index = self.pupil.build(optic, position=pupil_index, replace_existing=replace_existing)
        lens_front_index = self.lens_front.build(optic, position=pupil_index + 1, replace_existing=replace_existing)
        lens_back_index = self.lens_back.build(optic, position=lens_front_index + 1, replace_existing=replace_existing)
        self.retina.build(optic, position=lens_back_index + 1, replace_existing=replace_existing)

        # Sanity checks
        if optic.surface_group.stop_index != pupil_index:
            message = "The pupil is not located at the stop position."
            raise ValueError(message)

        if optic.image_surface != self.retina.surface:
            message = "The retina is not located at the image position."
            raise ValueError(message)
