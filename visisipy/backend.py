"""Backends for optical simulations.

This module provides a unified interface for different optical simulation backends,
as well as functions to interact with these backends.

Interfaces:

- `BaseAnalysisRegistry`: Base class for the backend analysis registry.
- `BaseBackend`: Base class for simulation backends.

Functions:

- `set_backend`: Set the backend to use for optical simulations.
- `get_backend`: Get the current backend, or initialize the default backend if not set.
- `get_oss`: Get the OpticStudioSystem instance if the current backend is OpticStudio.
- `get_optic`: Get the Optic instance if the current backend is Optiland.
- `update_settings`: Update settings on the current backend.

See Also
--------
visispy.opticstudio.backend : Backend for OpticStudio.
visispy.optiland.backend : Backend for Optiland.
"""

from __future__ import annotations

import json
import platform
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from inspect import get_annotations
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, TypeVar, cast, overload
from warnings import warn
from weakref import WeakValueDictionary

from visisipy.types import NotRequired, TypedDict, Unpack, ZernikeUnit

if TYPE_CHECKING:
    from os import PathLike

    from optiland.optic import Optic
    from pandas import DataFrame
    from zospy.zpcore import OpticStudioSystem

    from visisipy.analysis.cardinal_points import CardinalPointsResult
    from visisipy.models import BaseEye, EyeModel
    from visisipy.opticstudio.backend import OpticStudioSettings
    from visisipy.optiland.backend import OptilandSettings
    from visisipy.refraction import FourierPowerVectorRefraction
    from visisipy.types import ApertureType, FieldCoordinate, FieldType, SampleSize
    from visisipy.wavefront import ZernikeCoefficients

__all__ = (
    "DEFAULT_BACKEND_SETTINGS",
    "BackendAccessError",
    "BackendSettings",
    "BackendType",
    "BaseBackend",
    "get_backend",
    "get_optic",
    "get_oss",
    "save_model",
    "set_backend",
    "update_settings",
)


_BACKEND: BaseBackend | None = None

BackendType = Literal["opticstudio", "optiland"]
_DEFAULT_BACKEND: BackendType = "opticstudio" if platform.system() == "Windows" else "optiland"


_Analysis = TypeVar("_Analysis", bound=Callable)


class _AnalysisMethod(Generic[_Analysis]):
    def __init__(self, analysis) -> None:
        self._analysis = analysis

    @overload
    def __get__(self, instance: None, owner: type[BaseAnalysisRegistry]) -> _AnalysisMethod[_Analysis]: ...

    @overload
    def __get__(self, instance: BaseAnalysisRegistry, owner: type[BaseAnalysisRegistry]) -> _Analysis: ...

    def __get__(
        self, instance: BaseAnalysisRegistry | None, owner: type[BaseAnalysisRegistry]
    ) -> _AnalysisMethod | _Analysis:
        if instance is None:
            return self

        return MethodType(self._analysis, instance.backend)


class _classproperty(property):  # noqa: N801
    def __get__(self, instance, owner=None):
        return self.fget(owner)


class BaseAnalysisRegistry(ABC):
    """Base class for analysis registry.

    Interface for the analysis methods of the backend. Backends should implement this interface
    including all the analysis methods. If an analysis method is not implemented in the backend,
    it should raise a NotImplementedError.

    Attributes
    ----------
    backend : BaseBackend
        The backend in which the analysis is performed.

    Methods
    -------
    cardinal_points(surface_1, surface_2)
        Calculate the cardinal points of the optical system.
    fft_psf(field_coordinate, wavelength, field_type, sampling)
        Calculate the FFT Point Spread Function (PSF) at the retina surface.
    huygens_psf(field_coordinate, wavelength, field_type, pupil_sampling, image_sampling)
        Calculate the Huygens Point Spread Function (PSF) at the retina surface.
    raytrace(coordinates, wavelengths, field_type, pupil)
        Perform a raytrace through the optical system.
    refraction(field_coordinate, wavelength, sampling, pupil_diameter, field_type)
        Calculate the spherical equivalent of refraction for the optical system.
    strehl_ratio(field_coordinate, wavelength, field_type, sampling, psf_type)
        Calculate the Strehl ratio of the optical system.
    zernike_standard_coefficients(field_coordinate, wavelength, field_type, sampling, maximum_term)
        Calculate the Zernike standard coefficients for the optical system.
    """

    def __init__(self, backend: BaseBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> BaseBackend:
        return self._backend

    @abstractmethod
    def cardinal_points(
        self,
        surface_1: int | None = None,
        surface_2: int | None = None,
    ) -> tuple[CardinalPointsResult, Any]: ...

    @abstractmethod
    def fft_psf(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        sampling: SampleSize | str | int = 128,
    ) -> tuple[DataFrame, Any]: ...

    @abstractmethod
    def huygens_psf(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        pupil_sampling: SampleSize | str | int = 128,
        image_sampling: SampleSize | str | int = 128,
    ) -> tuple[DataFrame, Any]: ...

    @abstractmethod
    def opd_map(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        sampling: SampleSize | str | int = 128,
        *,
        remove_tilt: bool = True,
        use_exit_pupil_shape: bool = False,
    ) -> tuple[DataFrame, Any]: ...

    @abstractmethod
    def raytrace(
        self,
        coordinates: Sequence[FieldCoordinate] | None = None,
        wavelengths: Sequence[float] | None = None,
        field_type: FieldType = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> tuple[DataFrame, Any]: ...

    @abstractmethod
    def refraction(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        sampling: SampleSize | str | int = 64,
        pupil_diameter: float | None = None,
        field_type: FieldType = "angle",
        *,
        use_higher_order_aberrations: bool = True,
    ) -> tuple[FourierPowerVectorRefraction, Any]: ...

    @abstractmethod
    def strehl_ratio(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        sampling: SampleSize | str | int = 128,
        psf_type: Literal["fft", "huygens"] = "huygens",
    ) -> tuple[float, Any]: ...

    @abstractmethod
    def zernike_standard_coefficients(
        self,
        field_coordinate: FieldCoordinate | None = None,
        wavelength: float | None = None,
        field_type: FieldType = "angle",
        sampling: SampleSize | str | int = 64,
        maximum_term: int = 45,
        unit: ZernikeUnit = "microns",
    ) -> tuple[ZernikeCoefficients, Any]: ...


class BackendSettings(TypedDict, total=False):
    """A dictionary containing the settings for the backend."""

    field_type: FieldType
    """The field type to use in the optical system. Must be one of 'angle' or 'object_height'."""

    fields: Sequence[FieldCoordinate]
    """List of field coordinates to use in the optical system."""

    wavelengths: Sequence[float]
    """List of wavelengths to use in the optical system."""

    aperture_type: ApertureType
    """The aperture type to use in the optical system. Must be one of 'float_by_stop_size', 'entrance_pupil_diameter',
    'image_f_number', or 'object_numeric_aperture'.
    """

    aperture_value: NotRequired[float]
    """The aperture value to use in the optical system. Not required for 'float_by_stop_size'."""


DEFAULT_BACKEND_SETTINGS = BackendSettings(
    field_type="angle",
    fields=[(0, 0)],
    wavelengths=[0.543],
    aperture_type="float_by_stop_size",
    aperture_value=2,
)


_Settings = TypeVar("_Settings", bound=BackendSettings)


class BaseBackend(ABC, Generic[_Settings]):
    """Base class for optical simulation backends.

    Backends should implement this interface to provide a unified interface for optical simulations.
    """

    _instances: WeakValueDictionary[type[Self], Self] = WeakValueDictionary()

    def __new__(cls, *args, **kwargs) -> Self:  # noqa: ARG004
        """Store the first instance of each backend subclass to allow retrieving existing instances later."""
        instance = super().__new__(cls)

        if cls not in cls._instances:
            cls._instances[cls] = instance

        return instance

    @classmethod
    def get_instance(cls) -> Self | None:
        """Get the current instance of the backend.

        Returns
        -------
        BaseBackend | None
            The current instance of the backend, or None if it has not been initialized yet.

        """
        if cls not in cls._instances:
            return None

        return cast("Self", cls._instances[cls])

    @abstractmethod
    def __init__(self, **settings: Unpack[BackendSettings]) -> None: ...

    type: ClassVar[BackendType]
    _settings_type: type[_Settings]

    @property
    @abstractmethod
    def model(self) -> BaseEye | None: ...

    @property
    @abstractmethod
    def settings(self) -> _Settings: ...

    @property
    @abstractmethod
    def analysis(self) -> BaseAnalysisRegistry: ...

    @abstractmethod
    def update_settings(self, **settings: Unpack[BackendSettings]) -> None: ...

    @abstractmethod
    def build_model(
        self,
        model: EyeModel,
        *,
        start_from_index: int = 0,
        replace_existing: bool = False,
        object_distance: float = float("inf"),
        **kwargs,
    ) -> BaseEye: ...

    @abstractmethod
    def clear_model(self) -> None: ...

    @abstractmethod
    def save_model(self, filename: str | PathLike | None = None) -> None: ...

    @abstractmethod
    def load_model(self, filename: str | PathLike, *, apply_settings: bool = False) -> None: ...

    def validate_settings(self, name: str | _Settings | Sequence[str]) -> None:
        """Check if the backend has the specified setting.

        Parameters
        ----------
        name : str | BackendSettings | Sequence[str]
            The name or settings to check.

        Raises
        ------
        KeyError
            If the setting does not exist.
        TypeError
            If the name parameter is not a string, a settings dictionary, or a sequence of strings.
        """
        allowed_keys = get_annotations(self._settings_type).keys()

        if isinstance(name, str):
            if name not in allowed_keys:
                msg = f"Setting {name} is not a valid backend setting."
                raise KeyError(msg)
        elif isinstance(name, dict | Sequence):
            if not set(name).issubset(allowed_keys):
                invalid_keys = [key for key in name if key not in allowed_keys]
                msg = (
                    f"Setting {invalid_keys[0]} is not a valid backend setting."
                    if len(invalid_keys) == 1
                    else f"Settings {', '.join(invalid_keys)} are not valid backend settings."
                )
                raise KeyError(msg)
        else:
            raise TypeError("name must be a string, dictionary, or a sequence of strings.")

    def get_setting(self, name: str) -> Any:
        """Get a value from the backend settings.

        This method is mainly intended for internal use, to prevent the type checker from warning
        about potentially missing keys in the settings dictionary.

        Parameters
        ----------
        name : str
            The name of the setting to get.

        Returns
        -------
        Any
            The value of the setting.

        Raises
        ------
        KeyError
            If the setting does not exist.
        """
        self.validate_settings(name)

        if name not in self.settings:
            raise KeyError(f"Setting '{name}' has not been set.")

        return self.settings[name]

    def save_settings(self, filename: str | PathLike) -> None:
        if not str(filename).endswith(".json"):
            raise ValueError("Settings file must have a '.json' extension.")

        Path(filename).write_text(json.dumps(self.settings, indent=4, sort_keys=True), encoding="utf-8")


class BackendAccessError(RuntimeError):
    """Error raised when trying to access unavailable or non-initialized backend features."""


_BT = TypeVar("_BT", bound=BaseBackend)


def _get_or_initialize_backend(backend_type: type[_BT], settings: BackendSettings) -> _BT:
    """Get the current instance of the specified backend, or initialize it if it has not been initialized yet.

    Returns
    -------
    BaseBackend
        The current or new instance of the specified backend.
    """
    if instance := backend_type.get_instance():
        instance.update_settings(**settings)
        return instance

    return backend_type(**settings)


@overload
def set_backend(
    backend: Literal["opticstudio"],
    **settings: Unpack[OpticStudioSettings],
) -> None: ...


@overload
def set_backend(
    backend: Literal["optiland"],
    **settings: Unpack[OptilandSettings],
) -> None: ...


def set_backend(
    backend: BackendType = _DEFAULT_BACKEND,
    **settings: Any,
) -> None:
    """Set the backend to use for optical simulations.

    Parameters
    ----------
    backend : BackendType
        The backend to use. Must be one of {'opticstudio', 'optiland'}. Defaults to 'opticstudio' on Windows and 'optiland' elsewhere.
    settings : BackendSettings, optional
        Dictionary with settings for the backend. Defaults to `None`.

    Raises
    ------
    ValueError
        If an invalid backend is specified.
    """
    global _BACKEND  # noqa: PLW0603

    if _BACKEND is not None:
        warn(
            f"The backend is already set to {_BACKEND.__class__.__name__}. "
            f"Reconfiguring the backend is not recommended and may cause issues."
        )

    if backend == "opticstudio":
        from visisipy.opticstudio import OpticStudioBackend  # noqa: PLC0415

        _BACKEND = _get_or_initialize_backend(OpticStudioBackend, settings)  # type: ignore
    elif backend == "optiland":
        from visisipy.optiland import OptilandBackend  # noqa: PLC0415

        _BACKEND = _get_or_initialize_backend(OptilandBackend, settings)  # type: ignore
    else:
        raise ValueError(f"Unknown backend: {backend}")


def get_backend() -> BaseBackend:
    """Get the current backend.

    The backend is set to the default backend if it has not been set yet.

    Returns
    -------
    BaseBackend
        The current backend.
    """
    if _BACKEND is None:
        set_backend(_DEFAULT_BACKEND)

    return cast("BaseBackend", _BACKEND)


def get_oss() -> OpticStudioSystem:
    """Get the OpticStudioSystem instance from the current backend.

    Returns
    -------
    OpticStudioSystem
        The OpticStudioSystem instance if the OpticStudio backend is currently initialized.

    Raises
    ------
    BackendAccessError
        If the OpticStudioBackend is not currently initialized, or if the platform is not Windows.
    """
    if platform.system() != "Windows":
        raise BackendAccessError("The OpticStudio backend is only available on Windows.")

    from visisipy.opticstudio import OpticStudioBackend  # noqa: PLC0415

    if instance := OpticStudioBackend.get_instance():
        return instance.oss

    raise BackendAccessError("The OpticStudio backend has not been initialized.")


def get_optic() -> Optic:
    """Get the Optic instance from the current backend.

    Returns
    -------
    Optic
        The Optic instance if the Optiland backend is currently initialized.

    Raises
    ------
    BackendAccessError
        If the OptilandBackend is not currently initialized.
    """
    from visisipy.optiland import OptilandBackend  # noqa: PLC0415

    if instance := OptilandBackend.get_instance():
        return instance.optic

    raise BackendAccessError("The Optiland backend has not been initialized.")


def update_settings(backend: BaseBackend | None = None, **settings: Unpack[BackendSettings]):
    """Update settings on the current backend.

    Optionally, the backend can be manually specified. By default, the current backend is used.

    Parameters
    ----------
    backend : BaseBackend | None
        The backend to update. If `None`, the current backend is used.
    **settings : Unpack[BackendSettings]
        The settings to update. The keys and values should match the backend's configuration schema.
    """
    if backend is None:
        backend = get_backend()

    backend.update_settings(**settings)


def save_model(filename: str | PathLike | None = None) -> None:
    """Save the current model to a file.

    Parameters
    ----------
    filename : str | PathLike | None
        The filename to save the model to. If `None`, a default filename is used.
        The default filename depends on the backend.

    Raises
    ------
    BackendAccessError
        If no model is currently loaded in the backend.
    """
    backend = get_backend()

    if backend.model is None:
        raise BackendAccessError("No model is currently loaded in the backend.")

    backend.save_model(filename)


def load_model(filename: str | PathLike, *, apply_settings: bool = False) -> None:
    """Load a model from a file.

    The backend is automatically selected based on the file extension. For `.zmx` and `.zos` files,
    the OpticStudio backend is used. For `.json` files, the Optiland backend is used.

    Parameters
    ----------
    filename : str | PathLike
        The filename to load the model from.
    apply_settings : bool, optional
        If `True`, the currently configured backend settings will be applied after loading the model.

    Raises
    ------
    BackendAccessError
        If an OpticStudio file is specified on a non-Windows platform.
        If the model could not be loaded.
    ValueError
        If the file extension is not supported by any of the available backends.
    """
    filename = Path(filename)

    load_backend: BackendType

    if filename.suffix.lower() in {".zmx", ".zos"}:
        if platform.system() != "Windows":
            msg = f"Cannot load {filename.suffix}. The OpticStudio backend is only available on Windows."
            raise BackendAccessError(msg)
        load_backend = "opticstudio"
    elif filename.suffix.lower() == ".json":
        load_backend = "optiland"
    else:
        msg = f"File type {filename.suffix} is not supported by any of the available backends."
        raise ValueError(msg)

    current_backend: BackendType | None = getattr(_BACKEND, "type", None)

    if current_backend != load_backend:
        set_backend(load_backend)

    backend = get_backend()
    backend.load_model(filename, apply_settings=apply_settings)
