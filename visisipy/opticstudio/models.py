"""Build and manage eye models in OpticStudio."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from visisipy.models import BaseEye, EyeModel
from visisipy.opticstudio.surfaces import OpticStudioSurface, make_surface

if TYPE_CHECKING:
    from zospy.zpcore import OpticStudioSystem


__all__ = ("OpticStudioEye", "OpticStudioReverseEye")


class BaseOpticStudioEye(BaseEye):
    """Base class for OpticStudio eye models."""

    @abstractmethod
    def __init__(self, model: EyeModel): ...

    @property
    @abstractmethod
    def pupil(self) -> OpticStudioSurface:
        """Iris / pupil surface."""
        ...

    @abstractmethod
    def build(
        self,
        oss: OpticStudioSystem,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
    ):
        """Create the eye in OpticStudio.

        Create the eye model in the provided `OpticStudioSystem` `oss`, starting from `start_from_index`.
        The iris (pupil) is located at the STOP surface, and the retina at the IMAGE surface. For the other
        parts, new surfaces will be inserted by default. If `replace_existing` is set to `True`, existing
        surfaces will be overwritten.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the eye model is created.
        start_from_index : int
            Index at which the first  surface of the eye is located.
        replace_existing : bool
            If `True`, replaces existing surfaces instead of inserting new ones. Defaults to `False`.

        Raises
        ------
        AssertionError
            If the retina is not located at the IMAGE surface.
        """
        ...

    @property
    @abstractmethod
    def eye_model(self) -> EyeModel: ...

    @property
    def surfaces(self) -> dict[str, OpticStudioSurface]:
        """Dictionary with surface names as keys and surfaces as values."""
        return {k.lstrip("_"): v for k, v in self.__dict__.items() if isinstance(v, OpticStudioSurface)}

    def relink_surfaces(self, oss: OpticStudioSystem) -> bool:
        """Link surfaces to OpticStudio surfaces based on their comments.

        Attempt to re-link the surfaces of the eye to surfaces defined in OpticStudio, using the surface's comments.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem

        Returns
        -------
        bool
            `True` if all surfaces could be relinked, `False` otherwise.
        """
        result = [s.relink_surface(oss) for s in self.surfaces.values()]

        return all(result)


class OpticStudioEye(BaseOpticStudioEye):
    """Eye model in OpticStudio."""

    def __init__(self, eye_model: EyeModel) -> None:
        """Create a new OpticStudio eye model.

        Parameters
        ----------
        eye_model : EyeModel
            Eye model specification from which the OpticStudio eye model is created.
        """
        self._eye_model = eye_model

        self._cornea_front = make_surface(eye_model.geometry.cornea_front, eye_model.materials.cornea, "cornea front")
        self._cornea_back = make_surface(
            eye_model.geometry.cornea_back,
            eye_model.materials.aqueous,
            "cornea back / aqueous",
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
        """Eye model specification from which the OpticStudio eye model is created."""
        return self._eye_model

    @property
    def cornea_front(self) -> OpticStudioSurface:
        """Cornea front surface."""
        return self._cornea_front

    @cornea_front.setter
    def cornea_front(self, value: OpticStudioSurface) -> None:
        self._cornea_front = value

    @property
    def cornea_back(self) -> OpticStudioSurface:
        """Cornea back surface."""
        return self._cornea_back

    @cornea_back.setter
    def cornea_back(self, value: OpticStudioSurface) -> None:
        self._cornea_back = value

    @property
    def pupil(self) -> OpticStudioSurface:
        """Iris / pupil surface."""
        return self._pupil

    @pupil.setter
    def pupil(self, value: OpticStudioSurface) -> None:
        self._pupil = value

    @property
    def lens_front(self) -> OpticStudioSurface:
        """Lens front surface."""
        return self._lens_front

    @lens_front.setter
    def lens_front(self, value: OpticStudioSurface) -> None:
        self._lens_front = value

    @property
    def lens_back(self) -> OpticStudioSurface:
        """Lens back surface."""
        return self._lens_back

    @lens_back.setter
    def lens_back(self, value: OpticStudioSurface) -> None:
        self._lens_back = value

    @property
    def retina(self) -> OpticStudioSurface:
        """Retina surface."""
        return self._retina

    @retina.setter
    def retina(self, value: OpticStudioSurface) -> None:
        self._retina = value

    def build(
        self,
        oss: OpticStudioSystem,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
    ):
        """Create the eye in OpticStudio.

        Create the eye model in the provided `OpticStudioSystem` `oss`, starting from `start_from_index`.
        The iris (pupil) is located at the STOP surface, and the retina at the IMAGE surface. For the other
        parts, new surfaces will be inserted by default. If `replace_existing` is set to `True`, existing
        surfaces will be overwritten.

        Parameters
        ----------
        oss : zospy.zpcore.OpticStudioSystem
            OpticStudioSystem in which the eye model is created.
        start_from_index : int
            Index of the surface after which the eye model will be built. Because the pupil will be located at the stop surface,
            `start_from_index` must be smaller than the index of the stop surface.
        replace_existing : bool
            If `True`, replaces existing surfaces instead of inserting new ones. Defaults to `False`.
        object_distance : float, optional
            Distance from the object surface (or the surface before the eye model) to the eye model. Defaults to infinity.

        Raises
        ------
        ValueError
            If the pupil is not located at the stop position.
            If the retina is not located at the image surface.
        """
        if start_from_index >= oss.LDE.StopSurface:
            message = "'start_from_index' must be smaller than the index of the stop surface."
            raise ValueError(message)

        if object_distance != float("inf"):
            oss.LDE.GetSurfaceAt(start_from_index).Thickness = object_distance

        cornea_front_index = self.cornea_front.build(
            oss, position=start_from_index + 1, replace_existing=replace_existing
        )
        cornea_back_index = self.cornea_back.build(
            oss, position=cornea_front_index + 1, replace_existing=replace_existing
        )
        pupil_index = self.pupil.build(oss, position=cornea_back_index + 1, replace_existing=True)
        lens_front_index = self.lens_front.build(oss, position=pupil_index + 1, replace_existing=replace_existing)
        lens_back_index = self.lens_back.build(oss, position=lens_front_index + 1, replace_existing=replace_existing)
        self.retina.build(oss, position=lens_back_index + 1, replace_existing=True)

        # Sanity checks
        if not self.pupil.surface.IsStop:
            message = "The pupil is not located at the stop position."
            raise ValueError(message)

        if not self.retina.surface.IsImage:
            message = "The retina is not located at the image position."
            raise ValueError(message)


class OpticStudioReverseEye(BaseOpticStudioEye):  # pragma: no cover
    def __init__(self, eye_model: EyeModel):
        self._eye_model = eye_model

        self._retina = OpticStudioSurface(
            comment="retina / vitreous",
            radius=-1 * eye_model.geometry.retina_curvature,
            conic=eye_model.geometry.retina_asphericity,
            thickness=eye_model.geometry.vitreous_thickness,
            refractive_index=eye_model.materials.vitreous_refractive_index,
        )
        self._lens_back = OpticStudioSurface(
            comment="lens back",
            radius=-1 * eye_model.geometry.lens_back_curvature,
            conic=eye_model.geometry.lens_back_asphericity,
            thickness=eye_model.geometry.lens_thickness,
            refractive_index=eye_model.materials.lens_refractive_index,
        )
        self._lens_front = OpticStudioSurface(
            comment="lens front",
            radius=-1 * eye_model.geometry.lens_front_curvature,
            conic=-eye_model.geometry.lens_front_asphericity,
            refractive_index=eye_model.materials.aqueous_refractive_index,
        )
        self._iris = OpticStudioSurface(
            comment="iris",
            is_stop=True,
            refractive_index=eye_model.materials.aqueous_refractive_index,
            semi_diameter=eye_model.geometry.iris_radius,
        )
        self._aqueous = OpticStudioSurface(
            comment="aqueous",
            thickness=eye_model.geometry.anterior_chamber_depth,
            refractive_index=eye_model.materials.aqueous_refractive_index,
        )
        self._cornea_back = OpticStudioSurface(
            comment="cornea back",
            radius=-1 * eye_model.geometry.cornea_back_curvature,
            conic=eye_model.geometry.cornea_back_asphericity,
            thickness=eye_model.geometry.cornea_thickness,
            refractive_index=eye_model.materials.cornea_refractive_index,
        )
        self._cornea_front = OpticStudioSurface(
            comment="cornea front",
            radius=-1 * eye_model.geometry.cornea_front_curvature,
            conic=eye_model.geometry.cornea_front_asphericity,
        )

    @property
    def eye_model(self) -> EyeModel:
        return self._eye_model

    @property
    def cornea_front(self) -> OpticStudioSurface:
        return self._cornea_front

    @cornea_front.setter
    def cornea_front(self, value: OpticStudioSurface) -> None:
        self._cornea_front = value

    @property
    def cornea_back(self) -> OpticStudioSurface:
        return self._cornea_back

    @cornea_back.setter
    def cornea_back(self, value: OpticStudioSurface) -> None:
        self._cornea_back = value

    @property
    def aqueous(self) -> OpticStudioSurface:
        return self._aqueous

    @aqueous.setter
    def aqueous(self, value: OpticStudioSurface) -> None:
        self._aqueous = value

    @property
    def iris(self) -> OpticStudioSurface:
        return self._iris

    @iris.setter
    def iris(self, value: OpticStudioSurface) -> None:
        self._iris = value

    @property
    def lens_front(self) -> OpticStudioSurface:
        return self._lens_front

    @lens_front.setter
    def lens_front(self, value: OpticStudioSurface) -> None:
        self._lens_front = value

    @property
    def lens_back(self) -> OpticStudioSurface:
        return self._lens_back

    @lens_back.setter
    def lens_back(self, value: OpticStudioSurface) -> None:
        self._lens_back = value

    @property
    def retina(self) -> OpticStudioSurface:
        return self._retina

    @retina.setter
    def retina(self, value: OpticStudioSurface) -> None:
        self._retina = value

    def build(
        self,
        oss: OpticStudioSystem,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
    ):
        self.retina.build(oss, position=start_from_index, replace_existing=True)
        self.lens_back.build(oss, position=start_from_index + 1, replace_existing=replace_existing)
        self.lens_front.build(oss, position=start_from_index + 2, replace_existing=replace_existing)
        self.iris.build(oss, position=start_from_index + 3, replace_existing=True)
        self.aqueous.build(oss, position=start_from_index + 4, replace_existing=replace_existing)
        self.cornea_back.build(oss, position=start_from_index + 5, replace_existing=replace_existing)
        self.cornea_front.build(oss, position=start_from_index + 6, replace_existing=replace_existing)

        # Sanity checks
        if not self.retina.surface.IsObject:
            message = "The retina is not located at the object position."
            raise ValueError(message)
