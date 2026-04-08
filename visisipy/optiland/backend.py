"""Optiland backend for Visisipy."""

from __future__ import annotations

from importlib import import_module
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import optiland.backend
from optiland.fields.field_types.angle import AngleField
from optiland.fields.field_types.object_height import ObjectHeightField
from optiland.fileio import load_optiland_file, save_optiland_file
from optiland.optic import Optic

from visisipy.backend import (
    DEFAULT_BACKEND_SETTINGS,
    BackendAccessError,
    BackendSettings,
    BaseBackend,
    Unpack,
)
from visisipy.optiland.analysis import OptilandAnalysisRegistry
from visisipy.optiland.models import OptilandEye

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from os import PathLike

    from optiland._types import ApertureType as OptilandApertureType

    from visisipy import EyeModel
    from visisipy.types import ApertureType, FieldType, OptilandRayAimingType


__all__ = (
    "OptilandBackend",
    "OptilandSettings",
)


ComputationBackend = Literal["numpy", "torch"]
TorchDevice = Literal["cpu", "cuda"]
TorchPrecision = Literal["float32", "float64"]


class OptilandSettings(BackendSettings, total=False):
    """Backend settings that are specific to the Optiland backend."""

    computation_backend: ComputationBackend
    """Backend used for numerical operations. Must be one of 'numpy' or 'torch'. See
    https://optiland.readthedocs.io/en/latest/developers_guide/configurable_backend.html for more details.

    When using 'torch', the 'torch' package must be installed manually.
    """

    torch_device: TorchDevice
    """Device to use for torch backend. Must be one of 'cpu' or 'cuda'. Only used if computation_backend is 'torch'."""

    torch_precision: TorchPrecision
    """Precision to use for torch backend. Must be one of 'float32' or 'float64'.
    Only used if computation_backend is 'torch'.
    """

    torch_use_grad_mode: bool
    """Globally enable or disable autodifferentiation for the 'torch' backend. Only used if computation_backend is
    'torch'.
    """

    ray_aiming: OptilandRayAimingType
    """The ray aiming method to be used in the optic. Must be one of 'paraxial', 'robust', or 'iterative'."""

    ray_aiming_max_iterations: int
    """The maximum number of iterations for the 'iterative' ray aiming method. Only used if ray_aiming is set to
    'iterative'.
    """

    ray_aiming_tolerance: float
    """The tolerance for convergence for the 'iterative' ray aiming method. Only used if ray_aiming is set to
    'iterative'.
    """


OPTILAND_DEFAULT_SETTINGS: OptilandSettings = {
    **DEFAULT_BACKEND_SETTINGS,
    "computation_backend": "numpy",
    "torch_device": "cpu",
    "torch_precision": "float64",
    "torch_use_grad_mode": False,
    "ray_aiming": "paraxial",
    "ray_aiming_max_iterations": 10,
    "ray_aiming_tolerance": 1e-6,
}
"""Default settings for the Optiland backend."""

OPTILAND_APERTURES: dict[ApertureType, OptilandApertureType] = {
    "float_by_stop_size": "float_by_stop_size",
    "entrance_pupil_diameter": "EPD",
    "image_f_number": "imageFNO",
    "object_numeric_aperture": "objectNA",
}


class OptilandBackend(BaseBackend[OptilandSettings]):
    """Optiland backend."""

    def __init__(self, **settings: Unpack[OptilandSettings]):
        """Initialize the Optiland backend.

        This method initializes the Optiland backend with the given settings and creates a new model.

        Parameters
        ----------
        settings : OptilandSettings | None, optional
            The settings to be used for the Optiland backend. If None, the default settings are used.
        """
        self._model = None
        self._settings = OptilandSettings(**OPTILAND_DEFAULT_SETTINGS)
        self._analysis = OptilandAnalysisRegistry(self)

        if len(settings) > 0:
            self.validate_settings(settings)
            self.settings.update(settings)

        self.new_model()
        super().__init__()

    type = "optiland"
    _settings_type = OptilandSettings

    @property
    def optic(self) -> Optic:
        if self._optic is None:
            raise BackendAccessError("The Optiland backend has not been initialized.")

        return self._optic

    @property
    def model(self) -> OptilandEye | None:
        return self._model

    @model.setter
    def model(self, value: OptilandEye | None) -> None:
        self._model = value

    @property
    def settings(self) -> OptilandSettings:
        return self._settings

    @property
    def analysis(self) -> OptilandAnalysisRegistry:
        return self._analysis

    def _apply_settings(self) -> None:
        self.set_aperture()
        self.set_fields(
            field_type=self.get_setting("field_type"),
            coordinates=self.get_setting("fields"),
        )
        self.set_wavelengths(self.get_setting("wavelengths"))
        self.set_computation_backend(
            self.get_setting("computation_backend"),
            torch_device=self.get_setting("torch_device"),
            torch_precision=self.get_setting("torch_precision"),
            torch_use_grad_mode=self.get_setting("torch_use_grad_mode"),
        )
        cls.set_ray_aiming(
            ray_aiming=cls.get_setting("ray_aiming"),
            ray_aiming_max_iterations=cls.get_setting("ray_aiming_max_iterations"),
            ray_aiming_tolerance=cls.get_setting("ray_aiming_tolerance"),
        )

    def update_settings(self, **settings: Unpack[OptilandSettings]) -> None:
        """Apply the provided settings to the Optiland backend.

        This method applies the provided settings to the Optiland backend.
        """
        if len(settings) > 0:
            self.validate_settings(settings)
            self.settings.update(settings)

        self._apply_settings()

    def new_model(
        self,
        *,
        save_old_model: bool = False,
        save_filename: PathLike | str | None = None,
    ) -> None:
        """Initialize a new optical system.

        This method initializes a new, empty optical system.
        """
        if save_old_model:
            self.save_model(save_filename)

        self._optic = Optic()

        self.update_settings()

    def build_model(
        self,
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
        if not replace_existing and self.model is not None:
            self.new_model()

        optiland_eye = OptilandEye(model)
        optiland_eye.build(
            self.optic,
            start_from_index=start_from_index,
            replace_existing=replace_existing,
            object_distance=object_distance,
            **kwargs,
        )

        # Update the aperture settings based on the model's pupil size if the aperture type is 'float_by_stop_size'.
        if self.get_setting("aperture_type") == "float_by_stop_size":
            self.update_settings(aperture_value=model.geometry.pupil.semi_diameter * 2)

        self.model = optiland_eye
        return optiland_eye

    def clear_model(self) -> None:
        """Clear the current optical system model.

        This method initializes a new optical system, discarding any existing model.
        """
        self.model = None
        self.new_model()

    def save_model(self, path: str | PathLike | None = None) -> None:
        """Save the current optical system model.

        This method saves the current optical system model to the specified path. If no path is provided,
        it saves the model to the current working directory with the default name (model.json).

        Parameters
        ----------
        path : str | PathLike | None, optional
            The path where the model should be saved. If None, the model is saved in the current working directory.
            The file extension must be .json.

        Raises
        ------
        ValueError
            If the file extension of the provided path is not .json.
        """
        if path is None:
            path = "model.json"

        if not str(path).endswith(".json"):
            raise ValueError("filename must end in .json")

        save_optiland_file(self.optic, path)

    def load_model(self, filename: str | PathLike, *, apply_settings: bool = False) -> None:
        filename = Path(filename)

        if not filename.exists():
            msg = f"The specified file does not exist: {filename}"
            raise FileNotFoundError(msg)
        if filename.suffix.lower() != ".json":
            msg = f"File has extension {filename.suffix}, but only .json is supported."
            raise ValueError(msg)

        self.model = None
        optic = load_optiland_file(str(filename))
        self._optic = optic

        if apply_settings:
            self._apply_settings()

    def get_aperture(self) -> tuple[ApertureType, float]:
        """Get the current aperture type and value.

        Returns
        -------
        tuple[str, float]
            A tuple containing the aperture type and value.

        Raises
        ------
        ValueError
            If the aperture type in the optical system is not recognized.
        """
        optiland_aperture_type = self.optic.aperture.ap_type
        aperture_value: float = self.optic.aperture.value

        aperture_type = next(
            (k for k, v in OPTILAND_APERTURES.items() if v == optiland_aperture_type),
            None,
        )

        if aperture_type not in OPTILAND_APERTURES:
            raise ValueError(
                f"Invalid aperture type '{aperture_type}'. Must be one of {list(OPTILAND_APERTURES.keys())}."
            )

        return aperture_type, aperture_value

    def set_aperture(self):
        aperture_type = self.get_setting("aperture_type")
        aperture_value = self.get_setting("aperture_value")

        if aperture_type not in OPTILAND_APERTURES:
            raise ValueError(
                f"Invalid aperture type '{aperture_type}'. Must be one of {list(OPTILAND_APERTURES.keys())}."
            )

        if OPTILAND_APERTURES[aperture_type] is NotImplemented:
            raise NotImplementedError(f"Aperture type '{aperture_type}' is not implemented in Optiland.")

        self.optic.set_aperture(
            aperture_type=OPTILAND_APERTURES[aperture_type],
            value=aperture_value,
        )

    @classmethod
    def set_ray_aiming(
        cls, ray_aiming: OptilandRayAimingType, ray_aiming_max_iterations: int, ray_aiming_tolerance: float
    ) -> None:
        """Set the ray aiming method for the optic.

        Parameters
        ----------
        ray_aiming : OptilandRayAimingType
            The ray aiming method to be used in the optic. Must be one of 'paraxial', 'robust', or 'iterative'.
        ray_aiming_max_iterations : int
            The maximum number of iterations for the 'iterative' ray aiming method. Only used if ray_aiming is set to 'iterative'.
        ray_aiming_tolerance : float
            The tolerance for convergence for the 'iterative' ray aiming method. Only used if ray_aiming is set to 'iterative'.

        Raises
        ------
        ValueError
            If an invalid ray aiming method is provided.
            If ray_aiming_max_iterations is not a positive integer.
            If ray_aiming_tolerance is not a positive float.
        """

        if ray_aiming not in {"paraxial", "robust", "iterative"}:
            raise ValueError("ray_aiming must be one of 'paraxial', 'robust', or 'iterative'.")

        if ray_aiming_max_iterations <= 0:
            raise ValueError("ray_aiming_max_iterations must be a positive integer.")

        if ray_aiming_tolerance <= 0:
            raise ValueError("ray_aiming_tolerance must be a positive float.")

        cls.get_optic().set_ray_aiming(
            mode=ray_aiming, max_iter=ray_aiming_max_iterations, tolerance=ray_aiming_tolerance
        )
    
    def get_fields(self) -> list[tuple[float, float]]:
        """Get the fields in the optical system.

        Returns
        -------
        list[tuple[float, float]]
            List of field coordinates.
        """
        return [(f.x, f.y) for f in self.optic.fields.fields]

    def get_field_type(self) -> FieldType:
        """Get the current field type.

        Returns
        -------
        FieldType
            The current field type, either "angle" or "object_height".

        Raises
        ------
        ValueError
            If the field type in the optical system is not recognized.
        """
        optiland_field_type = self.optic.field_definition

        if isinstance(optiland_field_type, AngleField):
            return "angle"
        if isinstance(optiland_field_type, ObjectHeightField):
            return "object_height"

        raise ValueError("Unsupported field type in the optical system.")

    def set_field_type(self, field_type: FieldType):
        """Set the field type for the optical system.

        Parameters
        ----------
        field_type : FieldType
            The type of field to be used in the optical system. Can be either "angle" or "object_height".

        Raises
        ------
        ValueError
            If an invalid field type is provided.
        """
        if field_type not in {"angle", "object_height"}:
            raise ValueError("field_type must be either 'angle' or 'object_height'.")

        self.optic.set_field_type(field_type)

    def set_fields(
        self,
        coordinates: Iterable[tuple[float, float]],
        field_type: FieldType = "angle",
    ):
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
        # Remove all fields
        self.optic.fields.fields.clear()

        self.set_field_type(field_type)

        for field in coordinates:
            self.optic.add_field(y=field[1], x=field[0])

    def add_field(self, coordinate: tuple[float, float]) -> int:
        """Add a single field to the optical system.

        Parameters
        ----------
        coordinate : tuple[float, float]
            A tuple representing the coordinates for the field.

        Returns
        -------
        int
            The index of the newly added field.
        """
        self.optic.add_field(y=coordinate[1], x=coordinate[0])
        return len(self.optic.fields.fields) - 1

    def get_wavelengths(self) -> list[float]:
        """Get the wavelengths in the optical system.

        Returns
        -------
        list[float]
            List of wavelengths.
        """
        return [w.value for w in self.optic.wavelengths.wavelengths]

    def set_wavelengths(self, wavelengths: Sequence[float]):
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

        # Remove all wavelengths
        self.optic.wavelengths.wavelengths.clear()

        for wavelength in wavelengths:
            self.optic.add_wavelength(wavelength)

    def add_wavelength(self, wavelength: float) -> int:
        """Add a single wavelength to the optical system.

        Parameters
        ----------
        wavelength : float
            The wavelength to be added.

        Returns
        -------
        int
            The index of the newly added wavelength.
        """
        self.optic.add_wavelength(wavelength)
        return len(self.optic.wavelengths.wavelengths) - 1

    def iter_fields(self) -> Generator[tuple[int, tuple[float, float]], Any, None]:
        """Iterate over the fields in the optical system.

        Yields
        ------
        int
            Field index.
        tuple[float, float]
            Field X and Y coordinates.
        """
        for i, f in enumerate(self.optic.fields.fields):
            yield i, (f.x, f.y)

    def iter_wavelengths(self) -> Generator[tuple[int, float], Any, None]:
        """Iterate over the wavelengths in the optical system.

        Yields
        ------
        int
            Wavelength index.
        float
            Wavelength value.
        """
        for i, w in enumerate(self.optic.wavelengths.wavelengths):
            yield i, w.value

    @staticmethod
    def _torch_set_default_device(device: TorchDevice) -> None:
        """Set the default device for the torch backend.

        Parameters
        ----------
        device : TorchDevice
            The device to set as default. Must be one of 'cpu' or 'cuda'.
        """
        torch = import_module("torch")

        torch.set_default_device(device)

    def set_computation_backend(
        self,
        backend: ComputationBackend,
        *,
        torch_device: TorchDevice = "cpu",
        torch_precision: TorchPrecision = "float32",
        torch_use_grad_mode: bool = False,
    ) -> None:
        """Set the computation backend for Optiland.

        Parameters
        ----------
        backend : ComputationBackend
            The computation backend to use. Must be one of 'numpy' or 'torch'.
        torch_device : str, optional
            The device to use for the 'torch' backend. Must be one of 'cpu' or 'cuda'. Defaults to 'cpu'.
        torch_precision : str, optional
            The precision to use for the 'torch' backend. Must be one of 'float32' or 'float64'. Defaults to 'float32'.
        torch_use_grad_mode : bool, optional
            Whether to enable gradient mode for the 'torch' backend. Defaults to False.

        Raises
        ------
        ValueError
            If an invalid backend is provided.
            If an invalid torch_device is provided.
            If an invalid torch_precision is provided.
        """
        if backend not in {"numpy", "torch"}:
            raise ValueError("computation_backend must be either 'numpy' or 'torch'.")

        if torch_device not in {"cpu", "cuda"}:
            raise ValueError("torch_device must be either 'cpu' or 'cuda'.")

        if torch_precision not in {"float32", "float64"}:
            raise ValueError("torch_precision must be either 'float32' or 'float64'.")

        if optiland.backend.get_backend() != backend:
            optiland.backend.set_backend(backend)

        if backend == "torch":
            if optiland.backend.get_device() != torch_device:
                optiland.backend.set_device(torch_device)
                self._torch_set_default_device(torch_device)
            if optiland.backend.get_precision() != torch_precision:
                optiland.backend.set_precision(torch_precision)
            if torch_use_grad_mode and not optiland.backend.grad_mode.requires_grad:
                optiland.backend.grad_mode.enable()
            elif not torch_use_grad_mode and optiland.backend.grad_mode.requires_grad:
                optiland.backend.grad_mode.disable()

    def update_pupil(self, new_value: float) -> None:
        """Update the pupil size in the optical system.

        Parameters
        ----------
        new_value : float
            The new pupil size to be set.
        """
        aperture_type = self.optic.aperture.ap_type

        self.optic.set_aperture(
            aperture_type=aperture_type,
            value=new_value,
        )
