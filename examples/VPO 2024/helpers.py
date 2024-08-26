from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def get_ray_output_angle(
    df: pd.DataFrame,
    reference_point: tuple[float, float] = (0, 0),
    coordinate="y",
):
    """Calculate the output angle of a ray with respect to the optical axis and a reference point."""
    x0, y0 = reference_point
    x1, y1 = df.loc[len(df) - 1, "z"], df.loc[len(df) - 1, coordinate]

    return np.rad2deg(np.arctan2(y1 - y0, x1 - x0))


class InputOutputAngles(NamedTuple):
    input_angle_field: float
    output_angle_pupil: float
    output_angle_np2: float
    output_angle_retina_center: float = None
    output_angle_navarro_np2: float = None
    location_np2: float = None
    location_retina_center: float = None
    patient: int | str | None = None

    @classmethod
    def from_ray_trace_result(
        cls,
        raytrace_result: pd.DataFrame,
        np2: float,
        np2_navarro: float | None = None,
        retina_center: float | None = None,
        patient: int | None = None,
        coordinate="y",
    ) -> InputOutputAngles:
        return cls(
            input_angle_field=raytrace_result.field[0][1],
            output_angle_pupil=get_ray_output_angle(raytrace_result, reference_point=(0, 0), coordinate=coordinate),
            output_angle_np2=get_ray_output_angle(raytrace_result, reference_point=(np2, 0), coordinate=coordinate),
            output_angle_retina_center=(
                get_ray_output_angle(
                    raytrace_result,
                    reference_point=(retina_center, 0),
                    coordinate=coordinate,
                )
                if retina_center is not None
                else None
            ),
            output_angle_navarro_np2=(
                get_ray_output_angle(
                    raytrace_result,
                    reference_point=(np2_navarro, 0),
                    coordinate=coordinate,
                )
                if np2_navarro is not None
                else None
            ),
            location_np2=np2,
            location_retina_center=retina_center or None,
            patient=patient,
        )
