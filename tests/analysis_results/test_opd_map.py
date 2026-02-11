from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas.testing import assert_frame_equal

import visisipy

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


_OPD_MAP_INDEX_TYPES = {"header_type": float, "index_type": float}


@pytest.mark.parametrize(
    "field",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("opd_map_0_0", **_OPD_MAP_INDEX_TYPES)),
        pytest.param((0, 10), marks=pytest.mark.test_data("opd_map_0_10", **_OPD_MAP_INDEX_TYPES)),
        pytest.param((10, 0), marks=pytest.mark.test_data("opd_map_10_0", **_OPD_MAP_INDEX_TYPES)),
        pytest.param((10, 5), marks=pytest.mark.test_data("opd_map_10_5", **_OPD_MAP_INDEX_TYPES)),
    ],
)
def test_opd_map(field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame):
    result = visisipy.analysis.opd_map(
        model=result_test_model,
        field_coordinate=field,
        sampling=128,
        wavelength=0.543,
        remove_tilt=False,
        use_exit_pupil_shape=False,
        backend=configure_backend,
    )

    assert_frame_equal(expected_result, result, atol=1e-6, check_names=False)
