"""OpticStudio backend for Visisipy."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

import zospy as zp

from visisipy.backend import (
    DEFAULT_BACKEND_SETTINGS,
    BackendAccessError,
    BackendSettings,
    BaseBackend,
)
from visisipy.opticstudio.analysis import OpticStudioAnalysisRegistry
from visisipy.opticstudio.models import BaseOpticStudioEye, OpticStudioEye

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from os import PathLike

    from zospy.api import _ZOSAPI
    from zospy.zpcore import ZOS, OpticStudioSystem

    from visisipy import EyeModel
    from visisipy.types import ApertureType, FieldCoordinate, FieldType, NotRequired, Self, Unpack


__all__ = ("OpticStudioBackend", "OpticStudioSettings")


RayAimingType = Literal["off", "paraxial", "real"]


class OpticStudioSettings(BackendSettings, total=False):
    """Backend settings that are specific to the OpticStudio backend."""

    mode: Literal["standalone", "extension"]
    """ZOSPy connection mode. Must be one of 'standalone' or 'extension'."""

    zosapi_nethelper: NotRequired[str]
    """Path to the ZOSAPI_NetHelper.dll file. If not provided, the path is automatically determined."""

    opticstudio_directory: NotRequired[str]
    """Path to the OpticStudio installation directory. If not provided, the path is automatically determined."""

    ray_aiming: RayAimingType
    """The ray aiming method to be used in the optical system. Must be one of 'off', 'paraxial', or 'real'."""


OPTICSTUDIO_DEFAULT_SETTINGS: OpticStudioSettings = {
    **DEFAULT_BACKEND_SETTINGS,
    "mode": "standalone",
    "ray_aiming": "off",
}


class OpticStudioBackend(BaseBackend[OpticStudioSettings]):
    """OpticStudio backend."""

    def __new__(cls, *args, **kwargs) -> Self:  # noqa: ARG004
        """Create a new instance of the OpticStudio backend.

        If an instance of the OpticStudio backend already exists, a warning is raised and the existing instance is returned.

        Returns
        -------
        OpticStudioBackend
            A new instance of the OpticStudio backend, or the existing instance if it has already been initialized.
        """
        if instance := cls._instances.get(cls):
            warn(
                "An instance of the OpticStudio backend already exists. Returning the existing instance. "
                "Reinitializing the backend is not necessary and may cause issues.",
                stacklevel=2,
            )
            return instance

        instance = super().__new__(cls)
        instance.__initialized = False

        return instance

    def __init__(self, **settings: Unpack[OpticStudioSettings]) -> None:
        """Initialize the OpticStudio backend.

        This method connects to the OpticStudio backend and initializes a new optical system.

        Parameters
        ----------
        settings : OpticStudioSettings | None, optional
            The settings to be used for the OpticStudio backend. If None, the default settings are used.
        """
        if self.__initialized:
            return

        self._zos = None
        self._oss = None
        self._model = None
        self._settings = OpticStudioSettings(**OPTICSTUDIO_DEFAULT_SETTINGS)
        self._analysis = OpticStudioAnalysisRegistry(self)

        if len(settings) > 0:
            self.validate_settings(settings)
            self.settings.update(settings)

        self.connect()
        self.new_model()

        self.__initialized = True
        super().__init__()

    type = "opticstudio"
    _settings_type = OpticStudioSettings

    @property
    def zos(self) -> ZOS:
        """The ZOS instance for the OpticStudio backend.

        Returns
        -------
        ZOS
            The ZOS instance for the OpticStudio backend.

        Raises
        ------
        BackendAccessError
            If the ZOS instance is not initialized.
        """
        if self._zos is None:
            raise BackendAccessError("The OpticStudio backend has not been initialized.")

        return self._zos

    @property
    def oss(self) -> OpticStudioSystem:
        """The OpticStudio system instance for the OpticStudio backend.

        Returns
        -------
        OpticStudioSystem
            The OpticStudio system instance for the OpticStudio backend.

        Raises
        ------
        BackendAccessError
            If the OpticStudio system instance is not initialized.
        """
        if self._oss is None:
            raise BackendAccessError("The OpticStudio backend has not been initialized.")

        return self._oss

    @property
    def model(self) -> BaseOpticStudioEye | None:
        """The current optical system model for the OpticStudio backend."""
        return self._model

    @model.setter
    def model(self, value: BaseOpticStudioEye | None) -> None:
        self._model = value

    @property
    def settings(self) -> OpticStudioSettings:
        """The current settings for the OpticStudio backend."""
        return self._settings

    @property
    def analysis(self) -> OpticStudioAnalysisRegistry:
        """Analysis registry for the OpticStudio backend."""
        return self._analysis

    def _apply_settings(self) -> None:
        """Apply the currently configured settings to the OpticStudio backend."""
        self.set_aperture()
        self.set_fields(self.get_setting("fields"), field_type=self.get_setting("field_type"))
        self.set_ray_aiming(self.oss, self.get_setting("ray_aiming"))
        self.set_wavelengths(self.get_setting("wavelengths"))

    def update_settings(self, **settings: Unpack[OpticStudioSettings]) -> None:
        """Apply the provided settings to the OpticStudio backend.

        This method applies the provided settings to the OpticStudio backend.
        """
        if len(settings) > 0:
            self.validate_settings(settings)
            self.settings.update(settings)

        self._apply_settings()

    def new_model(
        self,
        *,
        save_old_model: bool = False,
    ) -> None:
        """Initialize a new optical system.

        This method initializes a new, empty optical system.
        """
        self.oss.new(saveifneeded=save_old_model)

        self.update_settings()

    def build_model(
        self,
        model: EyeModel,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ) -> OpticStudioEye:
        """Builds an optical system based on the provided eye model.

        This method creates an OpticStudioEye instance from the provided eye model and builds the optical system.
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
            Additional keyword arguments to be passed to the OpticStudioEye build method.

        Returns
        -------
        OpticStudioEye
            The built optical system model.
        """
        if not replace_existing and self.model is not None:
            self.new_model()

        opticstudio_eye = OpticStudioEye(model)
        opticstudio_eye.build(
            self.oss,
            start_from_index=start_from_index,
            replace_existing=replace_existing,
            object_distance=object_distance,
            **kwargs,
        )

        # Update the aperture settings based on the model's pupil size if the aperture type is 'float_by_stop_size'.
        if self.get_setting("aperture_type") == "float_by_stop_size":
            self.update_settings(aperture_value=model.geometry.pupil.semi_diameter * 2)

        self.model = opticstudio_eye

        return opticstudio_eye

    def clear_model(self) -> None:
        """Clear the current optical system model.

        This method initializes a new optical system, discarding any existing model.
        """
        self.oss.new(saveifneeded=False)
        self.model = None

    def save_model(self, filename: str | PathLike | None = None) -> None:
        """Save the current optical system model.

        This method saves the current optical system to the specified path. If no path is provided,
        it saves the model to the current working directory with the default name.

        Parameters
        ----------
        path : str | PathLike | None, optional
            The path where the model should be saved. If None, the model is saved in the current working directory.
        """
        if filename is not None:
            self.oss.save_as(filename)
        else:
            self.oss.save()

    def load_model(self, filename: str | PathLike, *, apply_settings: bool = False) -> None:
        """Load an optical system model from a file.

        This only loads the optical system into the backend. The model is not parsed to an `OpticStudioEye` or `EyeModel`
        instance. Furthermore, backend settings are not applied automatically after loading a model.

        Parameters
        ----------
        filename : str | PathLike
            The path to the file from which to load the model.
        apply_settings : bool, optional
            If `True`, the currently configured backend settings will be applied after loading the model.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file extension is not supported by the OpticStudio backend.
        """
        filename = Path(filename)

        if not filename.exists():
            msg = f"The specified file does not exist: {filename}"
            raise FileNotFoundError(msg)
        if filename.suffix.lower() not in {".zmx", ".zos"}:
            msg = f"File has extension {filename.suffix}, but only .zmx and .zos are supported."
            raise ValueError(msg)

        self.model = None
        self.oss.load(str(filename))

        if apply_settings:
            self._apply_settings()

    def connect(self) -> None:
        """Connect to OpticStudio if not already connected."""
        if self._zos is None:
            self._zos = zp.ZOS(
                zosapi_nethelper=self.settings.get("zosapi_nethelper"),
                opticstudio_directory=self.settings.get("opticstudio_directory"),
            )

        if self._oss is None:
            self._oss = self.zos.connect(self.get_setting("mode"))

    def disconnect(self) -> None:
        """Disconnects the OpticStudio backend.

        This method closes the current optical system, sets the system and ZOS instances to None,
        and disconnects the ZOS instance.
        """
        if self._oss is not None:
            self.oss.close()
            self._oss = None

        if self._zos is not None:
            self.zos.disconnect()
            self._zos = None

    def get_aperture(self) -> tuple[ApertureType, float]:
        """Get the current aperture type and value of the optical system.

        This method retrieves the current aperture type and value from the optical system.
        If the aperture type is `float_by_stop_size`, the diameter of the stop surface is returned as the aperture value.
        Note that the diameter, and not the semi-diameter, is returned, to be consistent with the other aperture types.
        For other aperture types, the aperture value is returned directly from the `SystemData`.

        Returns
        -------
        zp.constants.SystemData.ZemaxApertureType
            The current aperture type.
        """
        aperture_type: ApertureType

        if self.oss.SystemData.Aperture.ApertureType == zp.constants.SystemData.ZemaxApertureType.FloatByStopSize:
            aperture_type = "float_by_stop_size"
        elif (
            self.oss.SystemData.Aperture.ApertureType == zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
        ):
            aperture_type = "entrance_pupil_diameter"
        elif self.oss.SystemData.Aperture.ApertureType == zp.constants.SystemData.ZemaxApertureType.ImageSpaceFNum:
            aperture_type = "image_f_number"
        elif self.oss.SystemData.Aperture.ApertureType == zp.constants.SystemData.ZemaxApertureType.ObjectSpaceNA:
            aperture_type = "object_numeric_aperture"

        if aperture_type == "float_by_stop_size":
            aperture_value = self.oss.LDE.GetSurfaceAt(self.oss.LDE.StopSurface).SemiDiameter * 2
        else:
            aperture_value = self.oss.SystemData.Aperture.ApertureValue

        return aperture_type, aperture_value

    def _set_aperture_value(self) -> None:
        if self.settings.get("aperture_value") is not None:
            if self.get_setting("aperture_type") == "float_by_stop_size":
                self.oss.LDE.GetSurfaceAt(self.oss.LDE.StopSurface).SemiDiameter = (
                    self.get_setting("aperture_value") / 2
                )
            else:
                self.oss.SystemData.Aperture.ApertureValue = self.get_setting("aperture_value")

    def set_aperture(self):
        if self.get_setting("aperture_type") == "float_by_stop_size":
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.FloatByStopSize
            self._set_aperture_value()
        elif self.get_setting("aperture_type") == "entrance_pupil_diameter":
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
            self._set_aperture_value()
        elif self.get_setting("aperture_type") == "image_f_number":
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ImageSpaceFNum
            self._set_aperture_value()
        elif self.get_setting("aperture_type") == "object_numeric_aperture":
            self.oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ObjectSpaceNA
            self._set_aperture_value()
        else:
            raise ValueError(
                "aperture_type must be one of 'float_by_stop_size', 'entrance_pupil_diameter', "
                "'image_f_number', or 'object_numeric_aperture'."
            )

    def get_fields(self) -> list[FieldCoordinate]:
        """Get the fields in the optical system.

        Returns
        -------
        list[tuple[float, float]]
            The fields in the optical system as tuples of (x, y) coordinates.
        """
        fields = []

        for i in range(self.oss.SystemData.Fields.NumberOfFields):
            field = self.oss.SystemData.Fields.GetField(i + 1)
            fields.append((field.X, field.Y))

        return fields

    def get_field_type(self) -> FieldType:
        """Get the field type of the optical system.

        Returns
        -------
        FieldType
            The field type of the optical system, either "angle" or "object_height".

        Raises
        ------
        ValueError
            If the field type in the optical system is not supported.
        """
        field_type = self.oss.SystemData.Fields.GetFieldType()
        if field_type == zp.constants.SystemData.FieldType.Angle:
            return "angle"
        if field_type == zp.constants.SystemData.FieldType.ObjectHeight:
            return "object_height"

        raise ValueError("Unsupported field type in the optical system.")

    def set_field_type(self, field_type: FieldType) -> None:
        """Set the field type of the optical system.

        Parameters
        ----------
        field_type : FieldType
            The field type to set for the optical system, either "angle" or "object_height".

        Raises
        ------
        ValueError
            If `field_type` is not "angle" or "object_height".
        """
        if field_type == "angle":
            self.oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
        elif field_type == "object_height":
            self.oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.ObjectHeight)
        else:
            raise ValueError("field_type must be either 'angle' or 'object_height'.")

    @staticmethod
    def _set_fields(
        oss: OpticStudioSystem,
        coordinates: Iterable[tuple[float, float]],
    ) -> None:
        oss.SystemData.Fields.DeleteAllFields()

        for i, c in enumerate(coordinates):
            if i == 0:
                field = oss.SystemData.Fields.GetField(1)
                field.X, field.Y, field.Weight = float(c[0]), float(c[1]), 1
            else:
                oss.SystemData.Fields.AddField(float(c[0]), float(c[1]), 1)

    def set_fields(
        self,
        coordinates: Iterable[tuple[float, float]],
        field_type: FieldType = "angle",
    ) -> None:
        """Set the fields for the optical system.

        This method removes any existing fields and adds the new ones provided.

        Parameters
        ----------
        coordinates : Iterable[tuple[float, float]]
            An iterable of tuples representing the coordinates for the fields.
        field_type : FieldType, optional
            The type of field to be used in the optical system. Can be either "angle" or "object_height".
            Defaults to "angle".
        """
        self.set_field_type(field_type)
        self._set_fields(self.oss, coordinates)

    def add_field(
        self,
        coordinate: tuple[float, float],
    ) -> int:
        """Add a field to the optical system.

        Parameters
        ----------
        coordinate : tuple[float, float]
            The field coordinate to add.

        Returns
        -------
        int
            The field number of the added field.
        """
        new_field = self.oss.SystemData.Fields.AddField(float(coordinate[0]), float(coordinate[1]), 1)

        return new_field.FieldNumber

    def get_field_number(self, coordinate: tuple[float, float]) -> int | None:
        """Returns the field number for the given field coordinate.

        If the field coordinate is not found, `None` is returned.

        Parameters
        ----------
        coordinate : tuple[float, float]
            The field coordinate to find.

        Returns
        -------
        int | None
            The field number, or `None` if the field coordinate is not present.
        """
        for i in range(self.oss.SystemData.Fields.NumberOfFields):
            field = self.oss.SystemData.Fields.GetField(i + 1)

            if coordinate == (field.X, field.Y):
                return i + 1

        return None

    def get_wavelengths(self) -> list[float]:
        """Get the wavelengths in the optical system.

        Returns
        -------
        list[float]
            The wavelengths in the optical system.
        """

        return [
            self.oss.SystemData.Wavelengths.GetWavelength(i + 1).Wavelength
            for i in range(self.oss.SystemData.Wavelengths.NumberOfWavelengths)
        ]

    @staticmethod
    def _remove_wavelenghts(oss: OpticStudioSystem) -> None:
        while oss.SystemData.Wavelengths.NumberOfWavelengths > 0:
            oss.SystemData.Wavelengths.RemoveWavelength(1)

    def set_wavelengths(self, wavelengths: Sequence[float]) -> None:
        """Set the wavelengths for the optical system.

        This method removes any existing wavelengths and adds the new ones provided.
        The weight for each wavelength is set to 1.0.

        Parameters
        ----------
        wavelengths : Sequence[float]
            A sequence of wavelengths to be set for the optical system.

        Raises
        ------
        ValueError
            If no wavelengths are provided.
        """
        if len(wavelengths) == 0:
            raise ValueError("At least one wavelength must be provided.")

        self._remove_wavelenghts(self.oss)

        for w in wavelengths:
            self.oss.SystemData.Wavelengths.AddWavelength(Wavelength=w, Weight=1.0)

    def add_wavelength(self, wavelength: float) -> int:
        """Add a wavelength to the optical system.

        Parameters
        ----------
        wavelength : float
            The wavelength to add.

        Returns
        -------
        int
            The wavelength number of the added wavelength.
        """
        new_wavelength = self.oss.SystemData.Wavelengths.AddWavelength(Wavelength=wavelength, Weight=1.0)

        return new_wavelength.WavelengthNumber

    def get_wavelength_number(self, wavelength: float) -> int | None:
        """Returns the wavelength number for the given wavelength.

        If the wavelength is not found, `None` is returned.

        Parameters
        ----------
        wavelength : float
            The wavelength to find.

        Returns
        -------
        int | None
            The wavelength number, or `None` if the wavelength is not present.
        """
        for i in range(self.oss.SystemData.Wavelengths.NumberOfWavelengths):
            if self.oss.SystemData.Wavelengths.GetWavelength(i + 1).Wavelength == wavelength:
                return i + 1

        return None

    @staticmethod
    def set_ray_aiming(oss: OpticStudioSystem, ray_aiming: RayAimingType) -> None:
        if ray_aiming == "off":
            oss.SystemData.RayAiming.RayAiming = zp.constants.SystemData.RayAimingMethod.Off
        elif ray_aiming == "paraxial":
            oss.SystemData.RayAiming.RayAiming = zp.constants.SystemData.RayAimingMethod.Paraxial
        elif ray_aiming == "real":
            oss.SystemData.RayAiming.RayAiming = zp.constants.SystemData.RayAimingMethod.Real
        else:
            raise ValueError("ray_aiming must be either 'off', 'paraxial', or 'real'.")

    def iter_fields(self) -> Generator[tuple[int, _ZOSAPI.SystemData.IField], Any, None]:
        """Iterate over the fields in the optical system.

        Yields
        ------
        tuple[int, IField]
            A tuple containing the field number and the field object.
        """
        for i in range(self.oss.SystemData.Fields.NumberOfFields):
            field = self.oss.SystemData.Fields.GetField(i + 1)

            yield field.FieldNumber, field

    def iter_wavelengths(self) -> Generator[tuple[int, float], Any, None]:
        """Iterate over the wavelengths in the optical system.

        Yields
        ------
        tuple[int, float]
            A tuple containing the wavelength number and the wavelength value.
        """
        for i in range(self.oss.SystemData.Wavelengths.NumberOfWavelengths):
            yield i + 1, self.oss.SystemData.Wavelengths.GetWavelength(i + 1).Wavelength

    def update_pupil(self, new_value: float) -> None:
        """Update the pupil size in the optical system.

        This method updates the pupil size in the optical system to the new value provided.
        Which value is updated depends on the aperture type set in the optical system.
        For `float_by_stop_size`, the semi-diameter of the stop surface is updated; `new_value`
        is interpreted as the pupil diameter. For other aperture types, the aperture value in
        `SystemData` is updated.

        Parameters
        ----------
        new_value : float
            The new pupil size to be set.
        """
        if self.oss.SystemData.Aperture.ApertureType == zp.constants.SystemData.ZemaxApertureType.FloatByStopSize:
            stop_surface = self.oss.LDE.StopSurface
            self.oss.LDE.GetSurfaceAt(stop_surface).SemiDiameter = new_value / 2
        else:
            self.oss.SystemData.Aperture.ApertureValue = new_value
