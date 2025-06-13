from __future__ import annotations

import pytest

from visisipy import EyeModel


class TestFFTPSFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling,expectation",
        [
            (None, None, "angle", 32),
            (None, None, "object_height", 64),
            ((0, 0), 0.550, "angle", 128),
            ((10, 5), 0.400, "object_height", 256),
            ((5.5, 5.5), 0.550, "angle", 512),
        ],
    )
    def test_fft_psf(
        self,
        optiland_backend,
        optiland_analysis,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        assert optiland_analysis.fft_psf(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
        )
