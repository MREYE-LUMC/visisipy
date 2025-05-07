"""OpticStudio backend for Visisipy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from warnings import warn

import zospy as zp
from zospy.zpcore import OpticStudioSystem

from visisipy.backend import (
    BackendSettings,
    BaseBackend,
    FieldCoordinate,
    NotRequired,
    Unpack,
    _classproperty,
)
from visisipy.opticstudio.analysis import OpticStudioAnalysisRegistry
from visisipy.opticstudio.models import BaseOpticStudioEye, OpticStudioEye

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from os import PathLike

    from zospy.api import _ZOSAPI
    from zospy.zpcore import ZOS

    from visisipy import EyeModel


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
    "field_type": "angle",
    "fields": [(0, 0)],
    "wavelengths": [0.543],
    "aperture_type": "float_by_stop_size",
    "mode": "standalone",
    "ray_aiming": "off",
}


class OpticStudioBackend(BaseBackend):
    """OpticStudio backend."""

    zos: ZOS | None = None
    oss: OpticStudioSystem | None = None
    model: BaseOpticStudioEye | None = None
    settings: OpticStudioSettings = OpticStudioSettings(**OPTICSTUDIO_DEFAULT_SETTINGS)
    _analysis: OpticStudioAnalysisRegistry | None = None

    @_classproperty
    def analysis(cls) -> OpticStudioAnalysisRegistry:  # noqa: N805
        """Provides access to the `OpticStudioAnalysisRegistry` instance.

        This property provides access to the `OpticStudioAnalysisRegistry` instance for performing various analyses on the optical
        system.

        Returns
        -------
        OpticStudioAnalysisRegistry
            The `OpticStudioAnalysisRegistry` instance.

        Raises
        ------
        RuntimeError
            If the OpticStudio backend has not been initialized.
        """
        if cls.oss is None:
            raise RuntimeError("The opticstudio backend has not been initialized.")
        if cls._analysis is None:
            cls._analysis = OpticStudioAnalysisRegistry(cls)

        return cls._analysis

    @classmethod
    def initialize(
        cls,
        **settings: Unpack[OpticStudioSettings],
    ) -> None:
        """Initialize the OpticStudio backend.

        This method connects to the OpticStudio backend and initializes a new optical system.

        Parameters
        ----------
        settings : OpticStudioSettings | None, optional
            The settings to be used for the OpticStudio backend. If None, the default settings are used.
        """
        if len(settings) > 0:
            cls.settings.update(settings)

        if cls.zos is None:
            cls.zos = zp.ZOS(
                zosapi_nethelper=cls.settings.get("zosapi_nethelper"),
                opticstudio_directory=cls.settings.get("opticstudio_directory"),
            )
        else:
            warn(
                "The OpticStudio backend has already been initialized. "
                "Reinitializing the backend is not necessary and may cause issues."
            )

        if cls.oss is None:
            cls.oss = cls.zos.connect(cls.get_setting("mode"))

        cls.new_model()

    @classmethod
    def update_settings(cls, **settings: Unpack[OpticStudioSettings]) -> None:
        """Apply the provided settings to the OpticStudio backend.

        This method applies the provided settings to the OpticStudio backend.
        """
        if len(settings) > 0:
            cls.settings.update(settings)

        if cls.oss is None:
            warn(
                "The OpticStudio backend settings can only be applied after initialization. "
                "Settings will be applied when the backend is initialized."
            )
        else:
            cls.set_aperture()
            cls.set_fields(cls.get_setting("fields"), field_type=cls.get_setting("field_type"))
            cls.set_ray_aiming(cls.oss, cls.get_setting("ray_aiming"))
            cls.set_wavelengths(cls.get_setting("wavelengths"))

    @classmethod
    def new_model(
        cls,
        *,
        save_old_model: bool = False,
    ) -> None:
        """Initialize a new optical system.

        This method initializes a new, empty optical system.
        """
        cls.get_oss().new(saveifneeded=save_old_model)

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
        if not replace_existing and cls.model is not None:
            cls.new_model()

        opticstudio_eye = OpticStudioEye(model)
        opticstudio_eye.build(
            cls.get_oss(),
            start_from_index=start_from_index,
            replace_existing=replace_existing,
            object_distance=object_distance,
            **kwargs,
        )

        cls.model = opticstudio_eye

        return opticstudio_eye

    @classmethod
    def clear_model(cls) -> None:
        """Clear the current optical system model.

        This method initializes a new optical system, discarding any existing model.
        """
        cls.get_oss().new(saveifneeded=False)
        cls.model = None

    @classmethod
    def save_model(cls, path: str | PathLike | None = None) -> None:
        """Save the current optical system model.

        This method saves the current optical system to the specified path. If no path is provided,
        it saves the model to the current working directory with the default name.

        Parameters
        ----------
        path : str | PathLike | None, optional
            The path where the model should be saved. If None, the model is saved in the current working directory.
        """
        if path is not None:
            cls.get_oss().save_as(path)
        else:
            cls.get_oss().save()

    @classmethod
    def disconnect(cls) -> None:
        """Disconnects the OpticStudio backend.

        This method closes the current optical system, sets the system and ZOS instances to None,
        and disconnects the ZOS instance.
        """
        if cls.oss is not None:
            cls.oss.close()
            cls.oss = None

        if cls.zos is not None:
            cls.zos.disconnect()
            cls.zos = None

    @classmethod
    def get_oss(cls) -> OpticStudioSystem:
        """Returns the current optical system.

        Returns
        -------
        OpticStudioSystem
            The current optical system.

        Raises
        ------
        RuntimeError
            If the OpticStudio system is not initialized.
        """
        if cls.oss is None:
            raise RuntimeError("No OpticStudio system initialized. Please initialize the backend first.")

        return cast(OpticStudioSystem, cls.oss)

    @classmethod
    def _set_aperture_value(cls) -> None:
        if cls.settings.get("aperture_value") is not None:
            cls.get_oss().SystemData.Aperture.ApertureValue = cls.get_setting("aperture_value")

    @classmethod
    def set_aperture(cls):
        if cls.get_setting("aperture_type") == "float_by_stop_size":
            cls.get_oss().SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.FloatByStopSize
        elif cls.get_setting("aperture_type") == "entrance_pupil_diameter":
            cls.get_oss().SystemData.Aperture.ApertureType = (
                zp.constants.SystemData.ZemaxApertureType.EntrancePupilDiameter
            )
            cls._set_aperture_value()
        elif cls.get_setting("aperture_type") == "image_f_number":
            cls.get_oss().SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ImageSpaceFNum
            cls._set_aperture_value()
        elif cls.get_setting("aperture_type") == "object_numeric_aperture":
            cls.get_oss().SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.ObjectSpaceNA
            cls._set_aperture_value()
        else:
            raise ValueError(
                "aperture_type must be one of 'float_by_stop_size', 'entrance_pupil_diameter', "
                "'image_f_number', or 'object_numeric_aperture'."
            )

    @classmethod
    def get_fields(cls) -> list[FieldCoordinate]:
        """Get the fields in the optical system.

        Returns
        -------
        list[tuple[float, float]]
            The fields in the optical system as tuples of (x, y) coordinates.
        """
        fields = []

        for i in range(cls.get_oss().SystemData.Fields.NumberOfFields):
            field = cls.get_oss().SystemData.Fields.GetField(i + 1)
            fields.append((field.X, field.Y))

        return fields

    @staticmethod
    def _set_field_type(oss: OpticStudioSystem, field_type: Literal["angle", "object_height"]) -> None:
        if field_type == "angle":
            oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
        elif field_type == "object_height":
            oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.ObjectHeight)
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

    @classmethod
    def set_fields(
        cls,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object_height"] = "angle",
    ) -> None:
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
        cls._set_field_type(cls.get_oss(), field_type)
        cls._set_fields(cls.get_oss(), coordinates)

    @classmethod
    def get_wavelengths(cls) -> list[float]:
        """Get the wavelengths in the optical system.

        Returns
        -------
        list[float]
            The wavelengths in the optical system.
        """

        return [
            cls.get_oss().SystemData.Wavelengths.GetWavelength(i + 1).Wavelength
            for i in range(cls.get_oss().SystemData.Wavelengths.NumberOfWavelengths)
        ]

    @staticmethod
    def _remove_wavelenghts(oss: OpticStudioSystem) -> None:
        while oss.SystemData.Wavelengths.NumberOfWavelengths > 0:
            oss.SystemData.Wavelengths.RemoveWavelength(1)

    @classmethod
    def set_wavelengths(cls, wavelengths: Iterable[float]) -> None:
        """Set the wavelengths for the optical system.

        This method removes any existing wavelengths and adds the new ones provided.
        The weight for each wavelength is set to 1.0.

        Parameters
        ----------
        wavelengths : Iterable[float]
            An iterable of wavelengths to be set for the optical system.
        """
        cls._remove_wavelenghts(cls.get_oss())

        for w in wavelengths:
            cls.get_oss().SystemData.Wavelengths.AddWavelength(Wavelength=w, Weight=1.0)

    @classmethod
    def get_wavelength_number(cls, wavelength: float) -> int | None:
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
        for i in range(cls.get_oss().SystemData.Wavelengths.NumberOfWavelengths):
            if cls.get_oss().SystemData.Wavelengths.GetWavelength(i + 1).Wavelength == wavelength:
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

    @classmethod
    def iter_fields(cls) -> Generator[tuple[int, _ZOSAPI.SystemData.IField], Any, None]:
        """Iterate over the fields in the optical system."""
        for i in range(cls.get_oss().SystemData.Fields.NumberOfFields):
            field = cls.get_oss().SystemData.Fields.GetField(i + 1)

            yield field.FieldNumber, field

    @classmethod
    def iter_wavelengths(cls) -> Generator[tuple[int, float], Any, None]:
        """Iterate over the wavelengths in the optical system."""
        for i in range(cls.get_oss().SystemData.Wavelengths.NumberOfWavelengths):
            yield i + 1, cls.get_oss().SystemData.Wavelengths.GetWavelength(i + 1).Wavelength
