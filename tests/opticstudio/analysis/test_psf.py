from __future__ import annotations

import pytest

from tests.helpers import build_args
from visisipy import EyeModel

pytestmark = [pytest.mark.needs_opticstudio]


class TestFFTPSFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling",
        [
            (None, None, None, None),
            (None, None, "object_height", 64),
            ((0, 0), 0.550, "angle", 128),
            ((10, 5), 0.400, "object_height", 256),
            ((5.5, 5.5), 0.550, "angle", 512),
        ],
    )
    def test_fft_psf(
        self,
        opticstudio_backend,
        opticstudio_analysis,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            non_null_defaults={"field_type", "sampling"},
        )

        assert opticstudio_analysis.fft_psf(**args)


class TestHuygensPSFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,pupil_sampling,image_sampling",
        [
            (None, None, None, None, None),
            (None, None, "object_height", 32, 32),
            ((0, 0), 0.550, "angle", 64, 128),
            ((10, 5), 0.400, "object_height", 256, 128),
            ((5.5, 5.5), 0.550, "angle", 64, 64),
        ],
    )
    def test_fft_psf(
        self,
        opticstudio_backend,
        opticstudio_analysis,
        field_coordinate,
        wavelength,
        field_type,
        pupil_sampling,
        image_sampling,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            pupil_sampling=pupil_sampling,
            image_sampling=image_sampling,
            non_null_defaults={"field_type", "pupil_sampling", "image_sampling"},
        )

        assert opticstudio_analysis.huygens_psf(**args)
