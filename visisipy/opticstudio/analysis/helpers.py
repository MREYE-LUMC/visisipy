"""Helper functions for OpticStudio analyses."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Generator

    from visisipy.opticstudio.backend import OpticStudioBackend
    from visisipy.types import FieldType


def set_wavelength(
    backend: OpticStudioBackend,
    wavelength: float | None = None,
) -> int:
    """Set the wavelength in the OpticStudio backend.

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
    wavelength : float | None, optional
        The wavelength to set. If `None`, the first wavelength in the system will be used. Defaults to `None`.

    Returns
    -------
    int
        The wavelength number that was set.
    """
    wavelength_number = 1 if wavelength is None else backend.get_wavelength_number(wavelength)
    wavelength = cast("float", wavelength)

    if wavelength_number is None:
        warn(f"Wavelength {wavelength} not found. Adding it to the system.", stacklevel=2)
        wavelength_number = backend.add_wavelength(wavelength)

    return wavelength_number


def set_field(
    backend: OpticStudioBackend,
    field_coordinate: tuple[float, float] | None = None,
    field_type: FieldType = "angle",
) -> int:
    """Set the field coordinate in the OpticStudio backend.

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate to set. If `None`, the first field in the system will be used. Defaults to `None`.
    field_type : FieldType, optional
        The type of field to be used when setting the field coordinate. Defaults to "angle".

    Returns
    -------
    int
        The field number that was set.
    """
    field_number = 1 if field_coordinate is None else backend.get_field_number(field_coordinate)
    field_coordinate = cast("tuple[float, float]", field_coordinate)

    if field_type != (current_field_type := backend.get_field_type()):
        warn(
            f"Changing field type from {current_field_type} to {field_type}.",
            stacklevel=2,
        )
        backend.set_field_type(field_type)

    if field_number is None:
        warn(
            f"Field coordinate {field_coordinate} not found. Adding it to the system.",
            stacklevel=2,
        )
        field_number = backend.add_field(field_coordinate)

    return field_number


@contextmanager
def primary_wavelength(backend: OpticStudioBackend, wavelength: float) -> Generator[None, Any, None]:
    """Context manager to temporarily set a primary wavelength in the OpticStudio backend.

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
    wavelength : float
        The wavelength to set as primary, in μm.

    Yields
    ------
    None
        The context manager does not yield any value.
    """
    original_primary_wavelength = backend.get_primary_wavelength()

    backend.set_primary_wavelength(wavelength)

    try:
        yield
    finally:
        backend.set_primary_wavelength(original_primary_wavelength)
