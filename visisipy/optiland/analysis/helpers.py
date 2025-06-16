from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend
    from visisipy.types import FieldType

__all__ = ("set_field", "set_wavelength")


def set_wavelength(
    backend: type[OptilandBackend],
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
    else:
        backend.set_wavelengths([wavelength])

    return wavelength


def set_field(
    backend: type[OptilandBackend],
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
    field_type : str, optional
        The type of field to be used when setting the field coordinate. Defaults to "angle".

    Returns
    -------
    tuple[float, float]
        The normalized field coordinate of the field that was set.
    """
    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    return backend.get_optic().fields.get_field_coords()[0]
