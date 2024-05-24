from __future__ import annotations

from os import PathLike
from typing import Iterable, TYPE_CHECKING, Literal

import zospy as zp
from zospy.zpcore import ZOS, OpticStudioSystem

from visisipy._backend import BaseBackend, _classproperty
from visisipy.opticstudio.analysis import OpticStudioAnalysis
from visisipy.opticstudio.models import BaseOpticStudioEye, OpticStudioEye

if TYPE_CHECKING:
    from visisipy import EyeModel


def initialize_opticstudio(
    mode: Literal["standalone", "extension"] = "standalone",
    zosapi_nethelper: str | None = None,
    opticstudio_directory: str | None = None,
    ray_aiming: Literal["off", "paraxial", "real"] = "off",
) -> tuple[ZOS, OpticStudioSystem]:
    zos = zp.ZOS(
        zosapi_nethelper=zosapi_nethelper, opticstudio_directory=opticstudio_directory
    )
    oss = zos.connect(mode)

    if ray_aiming == "off":
        oss.SystemData.RayAiming.RayAiming = zp.constants.SystemData.RayAimingMethod.Off
    elif ray_aiming == "paraxial":
        oss.SystemData.RayAiming.RayAiming = (
            zp.constants.SystemData.RayAimingMethod.Paraxial
        )
    elif ray_aiming == "real":
        oss.SystemData.RayAiming.RayAiming = (
            zp.constants.SystemData.RayAimingMethod.Real
        )
    else:
        raise ValueError("ray_aiming must be either 'off', 'paraxial', or 'real'.")

    oss.SystemData.Aperture.ApertureType = (
        zp.constants.SystemData.ZemaxApertureType.FloatByStopSize
    )

    return zos, oss


def _set_field_type(oss: OpticStudioSystem, field_type: str) -> None:
    if field_type == "angle":
        oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
    elif field_type == "object":
        oss.SystemData.Fields.SetFieldType(
            zp.constants.SystemData.FieldType.ObjectHeight
        )
    else:
        raise ValueError("field_type must be either 'angle' or 'object'.")


def _set_fields(
    oss: OpticStudioSystem,
    coordinates: Iterable[tuple[float, float]],
) -> None:
    oss.SystemData.Fields.DeleteAllFields()

    for i, c in enumerate(coordinates):
        if i == 0:
            field = oss.SystemData.Fields.GetField(1)
            field.X, field.Y, field.Weight = c[0], c[1], 1
        else:
            oss.SystemData.Fields.AddField(c[0], c[1], 1)


def _remove_wavelenghts(oss: OpticStudioSystem) -> None:
    while oss.SystemData.Wavelengths.NumberOfWavelengths > 0:
        oss.SystemData.Wavelengths.RemoveWavelength(1)


class OpticStudioBackend(BaseBackend):
    _zos: ZOS | None = None
    _oss: OpticStudioSystem | None = None
    _model: BaseOpticStudioEye | None = None

    @_classproperty
    def analysis(cls) -> OpticStudioAnalysis:
        """
        Provides access to the `OpticStudioAnalysis` instance.

        This property provides access to the `OpticStudioAnalysis` instance for performing various analyses on the optical
        system.

        Raises
        ------
        RuntimeError
            If the OpticStudio backend has not been initialized.

        Returns
        -------
        OpticStudioAnalysis
            The `OpticStudioAnalysis` instance.
        """
        if cls._oss is None:
            raise RuntimeError("The opticstudio backend has not been initialized.")

        return OpticStudioAnalysis(cls)

    @classmethod
    def initialize(
        cls,
        mode: Literal["standalone", "extension"] = "standalone",
        zosapi_nethelper: str | None = None,
        opticstudio_directory: str | None = None,
        ray_aiming: Literal["off", "paraxial", "real"] = "off",
    ) -> None:
        """
        Initializes the OpticStudio backend.

        This method connects to the OpticStudio backend and initializes a new optical system.

        Parameters
        ----------
        mode : Literal["standalone", "extension"], optional
            The mode to use when connecting to the OpticStudio backend. Can be either "standalone" or "extension".
            Defaults to "standalone".
        zosapi_nethelper : str | None, optional
            The path to the ZOS-API NetHelper DLL. If None, the path is determined automatically.
        opticstudio_directory : str | None, optional
            The path to the OpticStudio installation directory. If None, the path is determined automatically.
        ray_aiming : Literal["off", "paraxial", "real"], optional
            The ray aiming method to use. Can be either "off", "paraxial", or "real". Defaults to "off".
        """
        cls._zos, cls._oss = initialize_opticstudio(
            mode=mode,
            zosapi_nethelper=zosapi_nethelper,
            opticstudio_directory=opticstudio_directory,
            ray_aiming=ray_aiming,
        )

        cls._oss.new()

    @classmethod
    def build_model(
        cls, model: EyeModel, replace_existing: bool = False, **kwargs
    ) -> OpticStudioEye:
        """
        Builds an optical system based on the provided eye model.

        This method creates an OpticStudioEye instance from the provided eye model and builds the optical system.
        If `replace_existing` is True, any existing model is updated instead of building a completely new system.

        Parameters
        ----------
        model : EyeModel
            The eye model to be used for building the optical system model.
        replace_existing : bool, optional
            Whether to replace any existing model before building the new one. Defaults to False.
        **kwargs
            Additional keyword arguments to be passed to the OpticStudioEye build method.

        Returns
        -------
        OpticStudioEye
            The built optical system model.
        """
        if not replace_existing and cls._model is not None:
            cls.clear_model()

        opticstudio_eye = OpticStudioEye(model)
        opticstudio_eye.build(cls._oss, replace_existing=replace_existing, **kwargs)

        cls._model = opticstudio_eye

        return opticstudio_eye

    @classmethod
    def clear_model(cls) -> None:
        """
        Clears the current optical system model.

        This method initializes a new optical system, discarding any existing model.
        """
        cls._oss.new(saveifneeded=False)
        cls._model = None

    @classmethod
    def save_model(cls, path: str | PathLike | None = None) -> None:
        """
        Saves the current optical system model.

        This method saves the current optical system model to the specified path. If no path is provided,
        it saves the model to the current working directory with the default name.

        Parameters
        ----------
        path : str | PathLike | None, optional
            The path where the model should be saved. If None, the model is saved in the current working directory.
        """
        if path is not None:
            cls._oss.save_as(path)
        else:
            cls._oss.save()

    @classmethod
    def disconnect(cls) -> None:
        """
        Disconnects the OpticStudio backend.

        This method closes the current optical system, sets the system and ZOS instances to None,
        and disconnects the ZOS instance.
        """
        cls._oss.close()
        cls._oss = None
        cls._zos.disconnect()
        cls._zos = None

    @classmethod
    def set_fields(
        cls,
        coordinates: Iterable[tuple[float, float]],
        field_type: Literal["angle", "object_height"] = "angle",
    ) -> None:
        """
        Sets the fields for the optical system.

        This method removes any existing fields and adds the new ones provided.

        Parameters
        ----------
        coordinates : Iterable[tuple[float, float]]
            An iterable of tuples representing the coordinates for the fields.
        field_type : Literal["angle", "object_height"], optional
            The type of field to be used in the optical system. Can be either "angle" or "object_height".
            Defaults to "angle".
        """
        _set_field_type(cls._oss, field_type)
        _set_fields(cls._oss, coordinates)

    @classmethod
    def set_wavelengths(cls, wavelengths: Iterable[float]) -> None:
        """
        Sets the wavelengths for the optical system.

        This method removes any existing wavelengths and adds the new ones provided.
        The weight for each wavelength is set to 1.0.

        Parameters
        ----------
        wavelengths : Iterable[float]
            An iterable of wavelengths to be set for the optical system.
        """
        _remove_wavelenghts(cls._oss)

        for w in wavelengths:
            cls._oss.SystemData.Wavelengths.AddWavelength(Wavelength=w, Weight=1.0)

    @classmethod
    def get_wavelength_number(cls, wavelength: float) -> int | None:
        """Returns the wavelength number for the given wavelength.

        If the wavelength is not found, `None` is returned.

        Parameters
        ----------
        wavelength: float
            The wavelength to find.

        Returns
        -------
        int | None
            The wavelength number, or `None` if the wavelength is not present.
        """
        for i in range(cls._oss.SystemData.Wavelengths.NumberOfWavelengths):
            if cls._oss.SystemData.Wavelengths.GetWavelength(i + 1) == wavelength:
                return i + 1

        return None
