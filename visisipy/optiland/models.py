from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from optiland.surfaces import ImageSurface

from visisipy.models import BaseEye
from visisipy.optiland.surfaces import OptilandSurface, make_surface

if TYPE_CHECKING:
    from optiland.optic import Optic

    from visisipy import EyeModel


class OptilandEye(BaseEye):
    def __init__(self, eye_model: EyeModel) -> None:
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

    def build(self, optic: Optic, *, start_from_index: int = 0, replace_existing: bool = False) -> None:
        # Create an object surface if it does not exist
        if start_from_index == 0 and optic.surface_group.num_surfaces == 0:
            optic.surface_group.add_surface(index=start_from_index, replace_existing=replace_existing, thickness=np.inf)

        self.cornea_front.build(optic, position=start_from_index + 1, replace_existing=replace_existing)
        self.cornea_back.build(optic, position=start_from_index + 2, replace_existing=replace_existing)
        _pupil_index = start_from_index + 3
        self.pupil.build(optic, position=_pupil_index, replace_existing=replace_existing)
        self.lens_front.build(optic, position=start_from_index + 4, replace_existing=replace_existing)
        self.lens_back.build(optic, position=start_from_index + 5, replace_existing=replace_existing)
        self.retina.build(optic, position=start_from_index + 6, replace_existing=replace_existing)

        # Sanity checks
        if optic.surface_group.stop_index != _pupil_index:
            message = "The pupil is not located at the stop position."
            raise ValueError(message)

        if not optic.surface_group.surfaces[-1] == self.retina.surface:
            message = "The retina is not located at the image position."
            raise ValueError(message)  # noqa: TRY004
