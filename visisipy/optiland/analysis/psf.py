from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
from optiland.psf import FFTPSF

from visisipy.types import SampleSize

if TYPE_CHECKING:
    from visisipy.optiland import OptilandBackend
    from visisipy.types import FieldCoordinate, FieldType


def fft_psf(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 64,
    psf_type: Literal["linear", "logarithmic"] = "linear",
) -> tuple[pd.DataFrame, FFTPSF]:
    if psf_type == "linear":
        log = False
    elif psf_type == "logarithmic":
        log = True  # noqa: F841
    else:
        raise ValueError("Invalid PSF type. Must be 'linear' or 'logarithmic'.")

    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    if field_coordinate is not None:
        backend.set_fields([field_coordinate], field_type=field_type)

    if wavelength is None:
        wavelength = backend.get_wavelengths()[0]
    else:
        backend.set_wavelengths([wavelength])

    normalized_field = backend.get_optic().fields.get_field_coords()[0]

    psf = FFTPSF(
        optic=backend.get_optic(),
        field=normalized_field,
        wavelength=wavelength,
        num_rays=int(sampling),
        grid_size=int(2 * sampling),
    )

    df = pd.DataFrame(psf.psf / 100)  # Optiland returns a normalized PSF in range [0, 100

    return df, psf
