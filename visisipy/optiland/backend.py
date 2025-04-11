from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from optiland.fileio import save_optiland_file
from optiland.optic import Optic

from visisipy.backend import BackendSettings, BaseBackend, Unpack, _classproperty
from visisipy.optiland.analysis import OptilandAnalysisRegistry
from visisipy.optiland.models import OptilandEye

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from os import PathLike

    from visisipy import EyeModel


OPTILAND_DEFAULT_SETTINGS: BackendSettings = {
    "field_type": "angle",
    "fields": [(0, 0)],
    "wavelengths": [0.543],
    "aperture_type": "entrance_pupil_diameter",
    "aperture_value": 1,
}

OPTILAND_APERTURES = {
    "float_by_stop_size": NotImplemented,
    "entrance_pupil_diameter": "EPD",
    "image_f_number": "imageFNO",
    "object_numeric_aperture": "objectNA",
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
    def initialize(cls, **settings: Unpack[BackendSettings]) -> None:
        if len(settings) > 0:
            cls.settings.update(settings)

        cls.new_model()

    @classmethod
    def update_settings(cls, **settings: Unpack[BackendSettings]) -> None:
        if len(settings) > 0:
            cls.settings.update(settings)

        if cls.optic is not None:
            cls.set_aperture()
            cls.set_fields(
                field_type=cls.settings["field_type"],
                coordinates=cls.settings["fields"],
            )
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
    def build_model(
        cls, model: EyeModel, *, replace_existing: bool = False, **kwargs
    ) -> OptilandEye:
        if not replace_existing and cls.model is not None:
            cls.new_model()

        optiland_eye = OptilandEye(model)
        optiland_eye.build(cls.get_optic(), replace_existing=replace_existing, **kwargs)

        cls.model = optiland_eye
        return optiland_eye

    @classmethod
    def clear_model(cls) -> NotImplemented:
        cls.model = None
        cls.optic = None

    @classmethod
    def save_model(cls, filename: str | PathLike | None = None) -> None:
        if filename is None:
            filename = "model.json"

        if not str(filename).endswith(".json"):
            raise ValueError("filename must end in .json")

        save_optiland_file(cls.optic, filename)

    @classmethod
    def get_optic(cls) -> Optic:
        """Get the optic object.

        Returns
        -------
        Optic
            The optic object.

        Raises
        ------
        RuntimeError
            If the optic object is not initialized.
        """
        if cls.optic is None:
            raise RuntimeError(
                "No optic object initialized. Please initialize the backend first."
            )

        return cast(Optic, cls.optic)

    @classmethod
    def set_aperture(cls):

        # warn(cls.settings["aperture_type"])

        if cls.settings["aperture_type"] not in OPTILAND_APERTURES:
            raise ValueError(
                f"Invalid aperture type '{cls.settings['aperture_type']}'. "
                f"Must be one of {list(OPTILAND_APERTURES.keys())}."
            )

        if OPTILAND_APERTURES[cls.settings["aperture_type"]] is NotImplemented:
            raise NotImplementedError(
                f"Aperture type '{cls.settings['aperture_type']}' is not implemented in Optiland."
            )

        cls.get_optic().set_aperture(
            aperture_type=OPTILAND_APERTURES[cls.settings["aperture_type"]],
            value=cls.settings["aperture_value"],
        )

    @classmethod
    def get_fields(cls) -> list[tuple[float, float]]:
        """Get the fields in the optical system.

        Returns
        -------
        list[tuple[float, float]]
            List of field coordinates.
        """
        return [(f.x, f.y) for f in cls.get_optic().fields.fields]

    @classmethod
    def set_fields(
        cls,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object_height"] = "angle",
    ):
        if field_type not in {"angle", "object_height"}:
            raise ValueError(
                "field_type must be either 'angle' or 'object_height'."
            )

        # Remove all fields
        cls.get_optic().fields.fields.clear()

        cls.get_optic().set_field_type(field_type)

        for field in coordinates:
            cls.get_optic().add_field(y=field[1], x=field[0])

    @classmethod
    def get_wavelengths(cls) -> list[float]:
        """Get the wavelengths in the optical system.

        Returns
        -------
        list[float]
            List of wavelengths.
        """
        return [w.value for w in cls.get_optic().wavelengths.wavelengths]

    @classmethod
    def set_wavelengths(cls, wavelengths: Iterable[float]):
        # Remove all wavelengths
        cls.get_optic().wavelengths.wavelengths.clear()

        for wavelength in wavelengths:
            cls.get_optic().add_wavelength(wavelength)

    @classmethod
    def iter_fields(cls) -> Generator[tuple[int, tuple[float, float]], Any, None]:
        """Iterate over the fields in the optical system.

        Yields
        ------
        int
            Field index.
        tuple[float, float]
            Field X and Y coordinates.
        """
        for i, f in enumerate(cls.get_optic().fields.fields):
            yield i, (f.x, f.y)

    @classmethod
    def iter_wavelengths(cls) -> Generator[tuple[int, float], Any, None]:
        """Iterate over the wavelengths in the optical system.

        Yields
        ------
        int
            Wavelength index.
        float
            Wavelength value.
        """
        for i, w in enumerate(cls.get_optic().wavelengths.wavelengths):
            yield i, w.value
