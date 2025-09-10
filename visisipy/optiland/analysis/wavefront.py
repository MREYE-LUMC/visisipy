from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.wavefront import OPD
from pandas import DataFrame

from visisipy.optiland.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    from visisipy.optiland.backend import OptilandBackend


def opd_map(
    backend: type[OptilandBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    *,
    remove_tilt: bool = True,
) -> tuple[DataFrame, OPD]:
    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength = set_wavelength(backend, wavelength)
    normalized_field = set_field(backend, field_coordinate, field_type)

    opd_result = OPD(
        optic=backend.get_optic(),
        wavelength=wavelength,
        field=normalized_field,
        num_rays=int(sampling),
        distribution="uniform",
        strategy="chief_ray",
        remove_tilt=remove_tilt,
    )

    data = opd_result.generate_opd_map(num_points=int(sampling))

    opd_dataframe = DataFrame(
        data["z"],
        index=data["y"][:, 0],
        columns=data["x"][0, :],
        dtype=float,
    )

    return opd_dataframe, opd_result
