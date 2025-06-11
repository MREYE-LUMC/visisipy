from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visisipy.opticstudio.backend import OpticStudioBackend


def set_wavelength(
    backend: type[OpticStudioBackend],
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

    if wavelength_number is None:
        backend.set_wavelengths([wavelength])
        wavelength_number = 1

    return wavelength_number


def set_field(
    backend: type[OpticStudioBackend],
    field_coordinate: tuple[float, float] | None = None,
    field_type: str = "angle",
) -> int:
    """Set the field coordinate in the OpticStudio backend.

    Parameters
    ----------
    backend : OpticStudioBackend
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float] | None, optional
        The field coordinate to set. If `None`, the first field in the system will be used. Defaults to `None`.
    field_type : str, optional
        The type of field to be used when setting the field coordinate. Defaults to "angle".

    Returns
    -------
    int
        The field number that was set.
    """
    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    return 1
