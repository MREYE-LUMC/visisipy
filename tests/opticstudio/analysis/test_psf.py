from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import pytest

from visisipy import EyeModel

pytestmark = [pytest.mark.needs_opticstudio]


class TestFFTPSFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling,psf_type,expectation",
        [
            (None, None, "angle", 32, "linear", does_not_raise()),
            (None, None, "object_height", 64, "logarithmic", does_not_raise()),
            ((0, 0), 0.550, "angle", 128, "linear", does_not_raise()),
            ((10, 5), 0.400, "object_height", 256, "logarithmic", does_not_raise()),
            ((5.5, 5.5), 0.550, "angle", 512, "linear", does_not_raise()),
            (
                None,
                None,
                "object_height",
                64,
                "quadratic",
                pytest.raises(ValueError, match=re.escape("Invalid PSF type. Must be 'linear' or 'logarithmic'.")),
            ),
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
        psf_type,
        expectation,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        with expectation:
            assert opticstudio_analysis.fft_psf(
                field_coordinate=field_coordinate,
                wavelength=wavelength,
                field_type=field_type,
                sampling=sampling,
                psf_type=psf_type,
            )
