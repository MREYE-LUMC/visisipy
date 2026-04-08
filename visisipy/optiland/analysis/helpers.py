from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend
    from visisipy.types import FieldType

__all__ = ("set_field", "set_wavelength")


def set_wavelength(
    backend: OptilandBackend,
    wavelength: float | None = None,
) -> float:
    """Set the wavelength in the Optiland backend.

    Parameters
    ----------
    backend : OptilandBackend
        Reference to the Optiland backend.
    wavelength : float | None, optional
        The wavelength to set. If `None`, the first wavelength in the system will be used. Defaults to `None`.

    Returns
    -------
    float
        The wavelength that was set. If `None` was passed, the first wavelength in the system is returned.
    """
    if wavelength is None:
        wavelength = backend.get_wavelengths()[0]
    elif wavelength not in backend.get_wavelengths():
        warn(f"Wavelength {wavelength} not found. Adding it to the system.", stacklevel=2)
        backend.add_wavelength(wavelength)

    return wavelength


def set_field(
    backend: OptilandBackend,
    field_coordinate: tuple[float, float] | None = None,
    field_type: FieldType = "angle",
) -> tuple[float, float]:
    """Set the field coordinate in the Optiland backend.

    Parameters
    ----------
    backend : OptilandBackend
        Reference to the Optiland backend.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate to set. If `None`, the first field in the system will be used. Defaults to `None`.
    field_type : FieldType, optional
        The type of field to be used when setting the field coordinate. Defaults to "angle".

    Returns
    -------
    tuple[float, float]
        The normalized field coordinate of the field that was set.
    """
    if field_type != (current_field_type := backend.get_field_type()):
        warn(f"Changing field type from {current_field_type} to {field_type}.", stacklevel=2)
        backend.set_field_type(field_type)

    if field_coordinate is not None:
        try:
            field_index = backend.get_fields().index(field_coordinate)
        except ValueError:
            warn(f"Field coordinate {field_coordinate} not found. Adding it to the system.", stacklevel=2)
            field_index = backend.add_field(field_coordinate)
    else:
        field_index = 0

    field_x, field_y = backend.optic.fields.get_field_coords()[field_index]

    return float(field_x), float(field_y)
