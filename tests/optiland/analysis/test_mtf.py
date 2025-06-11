from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel


class TestFFTMTFAnalysis:
    @pytest.mark.parametrize(
        "coordinate,wavelength,field_type,sampling,maximum_frequency,expectation",
        [
            (None, None, None, None, None, does_not_raise()),
            ((0, 0), 0.550, None, None, None, does_not_raise()),
            ((0, 0), 0.550, "angle", None, None, does_not_raise()),
            ((1, 0), 0.632, "object_height", 64, None, does_not_raise()),
            ((0, 1), 0.543, "angle", 64, None, does_not_raise()),
            ((12, 34), 0.543, "angle", 64, 10, does_not_raise()),
        ],
    )
    def test_fft_mtf(
        self,
        coordinate,
        wavelength,
        field_type,
        sampling,
        maximum_frequency,
        expectation,
        optiland_backend,
        optiland_analysis,
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        args = build_args(
            non_null_defaults={"field_type", "sampling", "maximum_frequency"},
            field_coordinate=coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            maximum_frequency=maximum_frequency,
        )

        with expectation:
            assert optiland_analysis.fft_mtf(**args)

    def test_mtf_result_structure(self, optiland_backend, optiland_analysis):
        optiland_backend.build_model(EyeModel())

        result, _ = optiland_analysis.fft_mtf(
            field_coordinate=(0, 0), wavelength=0.550, field_type="angle", sampling=64
        )

        assert result.tangential.name == "Tangential MTF"
        assert result.sagittal.name == "Sagittal MTF"
        assert result.tangential.index.name == "frequency"
        assert result.sagittal.index.name == "frequency"
        assert result.tangential.index.min() == 0
        assert result.sagittal.index.min() == 0
