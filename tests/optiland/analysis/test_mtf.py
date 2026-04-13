from __future__ import annotations

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel


class TestFFTMTFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling",
        [
            ((0, 0), None, None, None),
            ((0, 0), 0.550, "angle", 64),
            ((0, 30), 0.550, "angle", 64),
            ((30, 0), 0.550, "angle", 64),
            ((10, 5), 0.400, "object_height", 128),
            ("all", 0.550, "angle", 64),
        ],
    )
    def test_fft_mtf(
        self,
        optiland_backend,
        optiland_analysis,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            non_null_defaults={"field_coordinate", "field_type", "sampling"},
        )

        result, _ = optiland_analysis.fft_mtf(**args)

        if field_coordinate == "all":
            assert set(result.keys()) == set(optiland_backend.settings["fields"])
        else:
            assert set(result.keys()) == {field_coordinate}

    def test_fft_mtf_all_returns_all_backend_fields(self, optiland_backend, optiland_analysis):
        fields = [(0, 0), (2, 0), (0, 2), (2, 2)]

        optiland_backend.update_settings(fields=fields)
        optiland_backend.build_model(EyeModel(), object_distance=float("inf"))

        result, _ = optiland_analysis.fft_mtf(field_coordinate="all")

        assert set(result.keys()) == set(fields)
