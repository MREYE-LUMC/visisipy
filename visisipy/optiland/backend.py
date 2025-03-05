from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from optiland.fileio import save_optiland_file
from optiland.optic import Optic

from visisipy.backend import BackendSettings, BaseBackend, _classproperty
from visisipy.optiland import OptilandEye
from visisipy.optiland.analysis import OptilandAnalysisRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable
    from os import PathLike

    from visisipy import EyeModel


OPTILAND_DEFAULT_SETTINGS: BackendSettings = {
    "field_type": "angle",
    "fields": [(0, 0)],
    "wavelengths": [0.543],
    "aperture_type": "entrance_pupil_diameter",
}


class OptilandBackend(BaseBackend):
    optic: Optic | None = None
    model: OptilandEye | None = None
    settings: BackendSettings = BackendSettings(**OPTILAND_DEFAULT_SETTINGS)
    _analysis: OptilandAnalysisRegistry | None = None

    @_classproperty
    def analysis(cls) -> OptilandAnalysisRegistry:  # noqa: N805
        if cls._analysis is None:
            cls._analysis = OptilandAnalysisRegistry(cls)

        return cls._analysis

    @classmethod
    def initialize(cls, *, settings: BackendSettings | None = None) -> None:
        if settings is not None:
            cls.settings.update(settings)

    @classmethod
    def update_settings(cls, *, settings: BackendSettings | None = None) -> None:
        if settings is not None:
            cls.settings.update(settings)

        if cls.optic is not None:
            cls.set_aperture()
            cls.set_fields(field_type=cls.settings["field_type"], coordinates=cls.settings["fields"])
            cls.set_wavelengths(cls.settings["wavelengths"])

    @classmethod
    def new_model(
        cls,
        *,
        save_old_model: bool = False,
        save_filename: PathLike | str | None = None,
    ) -> None:
        if save_old_model:
            cls.save_model(save_filename)

        cls.optic = Optic()

        cls.update_settings()

    @classmethod
    def build_model(cls, model: EyeModel, *, replace_existing: bool = False, **kwargs) -> NotImplemented:
        if not replace_existing and cls.model is not None:
            cls.new_model()

        optiland_eye = OptilandEye(model)
        optiland_eye.build(cls.optic, replace_existing=replace_existing, **kwargs)

        cls.model = optiland_eye
        return optiland_eye

    @classmethod
    def clear_model(cls) -> NotImplemented:
        return NotImplemented

    @classmethod
    def save_model(cls, filename: str | PathLike | None = None) -> None:
        if filename is None:
            filename = "model.json"

        if not str(filename).endswith(".json"):
            raise ValueError("filename must end in .json")

        save_optiland_file(cls.optic, filename)

    @classmethod
    def set_aperture(cls):
        optiland_apertures = {
            "float_by_stop_size": NotImplemented,
            "entrance_pupil_diameter": "EPD",
        }

        cls.optic.set_aperture(
            aperture_type=optiland_apertures[cls.settings["aperture_type"]],
            value=cls.settings["aperture_value"],
        )

    @classmethod
    def set_fields(
        cls,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object_height"] = "angle",
    ):
        # Remove all fields
        cls.optic.fields.fields.clear()

        cls.optic.set_field_type(field_type)

        for field in coordinates:
            cls.optic.add_field(y=field[1], x=field[0])

    @classmethod
    def set_wavelengths(cls, wavelengths: Iterable[float]):
        # Remove all wavelengths
        cls.optic.wavelengths.wavelengths.clear()

        for wavelength in wavelengths:
            cls.optic.add_wavelength(wavelength)

    @classmethod
    def iter_fields(cls):
        return NotImplemented

    @classmethod
    def iter_wavelengths(cls):
        return NotImplemented
