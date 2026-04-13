from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import visisipy

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


@pytest.mark.parametrize(
    "field",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("fft_mtf_0_0")),
        pytest.param((0, 10), marks=pytest.mark.test_data("fft_mtf_0_10")),
        pytest.param((10, 0), marks=pytest.mark.test_data("fft_mtf_10_0")),
    ],
)
def test_fft_mtf(field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame):
    configure_backend.update_settings(fields=[field])

    result = visisipy.analysis.fft_mtf(
        model=result_test_model,
        field_coordinate=field,
        field_type="angle",
        sampling=128,
        wavelength=0.543,
        backend=configure_backend,
    )

    assert np.allclose(result[field].tangential.index, expected_result["frequency_tangential"])
    assert np.allclose(result[field].sagittal.index, expected_result["frequency_sagittal"])
    assert np.allclose(result[field].tangential, expected_result["mtf_tangential"])
    assert np.allclose(result[field].sagittal, expected_result["mtf_sagittal"])
