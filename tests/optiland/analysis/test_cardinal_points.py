from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.models import EyeModel


class TestCardinalPointsAnalysis:
    @pytest.mark.parametrize(
        "surface_1,surface_2,expectation",
        [
            (None, None, does_not_raise()),
            (1, 6, does_not_raise()),
            (
                2,
                4,
                pytest.raises(
                    ValueError, match="Optiland only supports calculating cardinal points for the entire system"
                ),
            ),
            (
                -1,
                7,
                pytest.raises(
                    ValueError,
                    match="Optiland only supports calculating cardinal points for the entire system",
                ),
            ),
        ],
    )
    def test_cardinal_points(self, surface_1, surface_2, expectation, optiland_backend, optiland_analysis):
        optiland_backend.build_model(EyeModel())

        with expectation:
            assert optiland_analysis.cardinal_points(surface_1, surface_2)
