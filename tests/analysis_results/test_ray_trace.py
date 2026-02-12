from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import visisipy

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


RAY_TRACE_COORDINATES = [(x, y) for x in np.linspace(-60, 60, 5) for y in np.linspace(-60, 60, 5)]


@pytest.mark.parametrize(
    "pupil",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("ray_trace_pupil_0_0")),
        pytest.param((0, 1), marks=pytest.mark.test_data("ray_trace_pupil_0_1")),
        pytest.param((1, -1), marks=pytest.mark.test_data("ray_trace_pupil_1_-1")),
    ],
)
def test_raytrace(pupil, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame):
    result = visisipy.analysis.raytrace(
        model=result_test_model,
        coordinates=RAY_TRACE_COORDINATES,
        wavelengths=[0.543],
        field_type="angle",
        pupil=pupil,
        backend=configure_backend,
    )

    assert_frame_equal(expected_result[["wavelength", "x", "y", "z"]], result[["wavelength", "x", "y", "z"]], atol=1e-6)
