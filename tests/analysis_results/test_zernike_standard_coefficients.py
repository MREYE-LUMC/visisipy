from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

import visisipy

if TYPE_CHECKING:
    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


@pytest.mark.parametrize(
    "field",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("zernike_coefficients_0_0")),
        pytest.param((0, 10), marks=pytest.mark.test_data("zernike_coefficients_0_10")),
        pytest.param((10, 0), marks=pytest.mark.test_data("zernike_coefficients_10_0")),
        pytest.param((10, 5), marks=pytest.mark.test_data("zernike_coefficients_10_5")),
    ],
)
def test_zernike_standard_coefficients(
    field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame
):
    result = visisipy.analysis.zernike_standard_coefficients(
        model=result_test_model,
        field_coordinate=field,
        sampling=128,
        wavelength=0.543,
        field_type="angle",
        maximum_term=45,
        unit="microns",
        backend=configure_backend,
    )
    df = DataFrame(result.items(), columns=["coefficient", "value"])

    assert_frame_equal(expected_result, df, atol=1e-6)
