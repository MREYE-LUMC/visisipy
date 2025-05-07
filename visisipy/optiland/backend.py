"""Optiland backend for Visisipy."""

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
"""Default settings for the Optiland backend."""

OPTILAND_APERTURES = {
    "float_by_stop_size": NotImplemented,
    "entrance_pupil_diameter": "EPD",
    "image_f_number": "imageFNO",
    "object_numeric_aperture": "objectNA",
}


class OptilandBackend(BaseBackend):
    """Optiland backend."""

    optic: Optic | None = None
    model: OptilandEye | None = None
    settings: BackendSettings = BackendSettings(**OPTILAND_DEFAULT_SETTINGS)
    _analysis: OptilandAnalysisRegistry | None = None

    @_classproperty
    def analysis(cls) -> OptilandAnalysisRegistry:  # noqa: N805
        """Provides access to the `OptilandAnalysisRegistry` instance.

        This property provides access to the `OptilandAnalysisRegistry` instance for performing various analyses on the optical
        system.

        Returns
        -------
        OpticStudioAnalysisRegistry
            The `OptilandAnalysisRegistry` instance.

        Raises
        ------
        RuntimeError
            If the Optiland backend has not been initialized.
        """
        if cls._analysis is None:
            cls._analysis = OptilandAnalysisRegistry(cls)

        return cls._analysis

    @classmethod
    def initialize(cls, **settings: Unpack[BackendSettings]) -> None:
        """Initialize the Optiland backend.

        This method initializes the Optiland backend with the given settings and creates a new model.

        Parameters
        ----------
        settings : BackendSettings | None, optional
            The settings to be used for the Optiland backend. If None, the default settings are used.
        """
        if len(settings) > 0:
            cls.settings.update(settings)

        cls.new_model()

    @classmethod
    def update_settings(cls, **settings: Unpack[BackendSettings]) -> None:
        """Apply the provided settings to the Optiland backend.

        This method applies the provided settings to the Optiland backend.
        """
        if len(settings) > 0:
            cls.settings.update(settings)

        if cls.optic is not None:
            cls.set_aperture()
            cls.set_fields(
                field_type=cls.get_setting("field_type"),
                coordinates=cls.get_setting("fields"),
            )
            cls.set_wavelengths(cls.get_setting("wavelengths"))

    @classmethod
    def new_model(
        cls,
        *,
        save_old_model: bool = False,
        save_filename: PathLike | str | None = None,
    ) -> None:
        """Initialize a new optical system.

        This method initializes a new, empty optical system.
        """
        if save_old_model:
            cls.save_model(save_filename)

        cls.optic = Optic()

        cls.update_settings()

    @classmethod
    def build_model(
        cls,
        model: EyeModel,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ) -> OptilandEye:
        """Builds an optical system based on the provided eye model.

        This method creates an OptilandEye instance from the provided eye model and builds the optical system.
        If `replace_existing` is True, any existing model is updated instead of building a completely new system.

        Parameters
        ----------
        model : EyeModel
            The eye model to be used for building the optical system model.
        start_from_index : int, optional
            The index of the surface after which the eye model will be built. The cornea front surface will be located
            at `start_from_index + 1`.
        replace_existing : bool, optional
            Whether to replace any existing model before building the new one. Defaults to False.
        object_distance : float
            Distance between the cornea front and the surface preceding the eye model.
        **kwargs
            Additional keyword arguments to be passed to the OptilandEye build method.

        Returns
        -------
        OptilandEye
            The built optical system model.
        """
        if not replace_existing and cls.model is not None:
            cls.new_model()

        optiland_eye = OptilandEye(model)
        optiland_eye.build(
            cls.get_optic(),
            start_from_index=start_from_index,
            replace_existing=replace_existing,
            object_distance=object_distance,
            **kwargs,
        )

        cls.model = optiland_eye
        return optiland_eye

    @classmethod
    def clear_model(cls) -> None:
        """Clear the current optical system model.

        This method initializes a new optical system, discarding any existing model.
        """
        cls.model = None
        cls.new_model()

    @classmethod
    def save_model(cls, path: str | PathLike | None = None) -> None:
        """Save the current optical system model.

        This method saves the current optical system model to the specified path. If no path is provided,
        it saves the model to the current working directory with the default name (model.json).

        Parameters
        ----------
        path : str | PathLike | None, optional
            The path where the model should be saved. If None, the model is saved in the current working directory.
            The file extension must be .json.
        """
        if path is None:
            path = "model.json"

        if not str(path).endswith(".json"):
            raise ValueError("filename must end in .json")

        save_optiland_file(cls.optic, path)

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
            raise RuntimeError("No optic object initialized. Please initialize the backend first.")

        return cast(Optic, cls.optic)

    @classmethod
    def set_aperture(cls):
        aperture_type = cls.get_setting("aperture_type")
        aperture_value = cls.get_setting("aperture_value")

        if aperture_type not in OPTILAND_APERTURES:
            raise ValueError(
                f"Invalid aperture type '{aperture_type}'. Must be one of {list(OPTILAND_APERTURES.keys())}."
            )

        if OPTILAND_APERTURES[aperture_type] is NotImplemented:
            raise NotImplementedError(f"Aperture type '{aperture_type}' is not implemented in Optiland.")

        cls.get_optic().set_aperture(
            aperture_type=OPTILAND_APERTURES[aperture_type],
            value=aperture_value,
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
        """Set the fields for the optical system.

        This method removes any existing fields and adds the new ones provided.

        Parameters
        ----------
        coordinates : Iterable[tuple[float, float]]
            An iterable of tuples representing the coordinates for the fields.
        field_type : Literal["angle", "object_height"], optional
            The type of field to be used in the optical system. Can be either "angle" or "object_height".
            Defaults to "angle".
        """
        if field_type not in {"angle", "object_height"}:
            raise ValueError("field_type must be either 'angle' or 'object_height'.")

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
        """Set the wavelengths for the optical system.

        This method removes any existing wavelengths and adds the new ones provided.
        The weight for each wavelength is set to 1.0.

        Parameters
        ----------
        wavelengths : Iterable[float]
            An iterable of wavelengths to be set for the optical system.
        """
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
