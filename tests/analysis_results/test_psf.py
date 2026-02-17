from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas.testing import assert_frame_equal

import visisipy

if TYPE_CHECKING:
    from pandas import DataFrame

    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


_PSF_INDEX_TYPES = {"header_type": float, "index_type": float}


@pytest.mark.parametrize(
    "field",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("fft_psf_0_0", **_PSF_INDEX_TYPES)),
        pytest.param((0, 10), marks=pytest.mark.test_data("fft_psf_0_10", **_PSF_INDEX_TYPES)),
        pytest.param((10, 0), marks=pytest.mark.test_data("fft_psf_10_0", **_PSF_INDEX_TYPES)),
    ],
)
def test_fft_psf(field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame):
    configure_backend.update_settings(fields=[field])

    result = visisipy.analysis.fft_psf(
        model=result_test_model,
        field_coordinate=field,
        sampling=128,
        wavelength=0.543,
        backend=configure_backend,
    )

    assert_frame_equal(expected_result, result, atol=1e-6, check_names=False)


@pytest.mark.parametrize(
    "field",
    [
        pytest.param((0, 0), marks=pytest.mark.test_data("huygens_psf_0_0", **_PSF_INDEX_TYPES)),
        pytest.param((0, 10), marks=pytest.mark.test_data("huygens_psf_0_10", **_PSF_INDEX_TYPES)),
        pytest.param((10, 0), marks=pytest.mark.test_data("huygens_psf_10_0", **_PSF_INDEX_TYPES)),
    ],
)
def test_huygens_psf(
    field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame
):
    configure_backend.update_settings(fields=[field])

    result = visisipy.analysis.huygens_psf(
        model=result_test_model,
        field_coordinate=field,
        pupil_sampling=128,
        image_sampling=128,
        wavelength=0.543,
        backend=configure_backend,
    )

    assert_frame_equal(expected_result, result, atol=1e-6, check_names=False)
