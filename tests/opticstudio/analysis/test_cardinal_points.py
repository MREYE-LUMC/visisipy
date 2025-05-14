from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.models import EyeModel

pytestmark = [pytest.mark.needs_opticstudio]


class TestCardinalPointsAnalysis:
    @pytest.mark.parametrize(
        "surface_1,surface_2,expectation",
        [
            (None, None, does_not_raise()),
            (1, 6, does_not_raise()),
            (2, 4, does_not_raise()),
            (
                -1,
                7,
                pytest.raises(
                    ValueError,
                    match="surface_1 and surface_2 must be between 1 and the number of surfaces in the system",
                ),
            ),
            (
                1,
                8,
                pytest.raises(
                    ValueError,
                    match="surface_1 and surface_2 must be between 1 and the number of surfaces in the system",
                ),
            ),
            (3, 3, pytest.raises(ValueError, match="surface_1 must be less than surface_2")),
            (4, 2, pytest.raises(ValueError, match="surface_1 must be less than surface_2")),
        ],
    )
    def test_cardinal_points(self, opticstudio_backend, opticstudio_analysis, surface_1, surface_2, expectation):
        opticstudio_backend.build_model(EyeModel())

        with expectation:
            assert opticstudio_analysis.cardinal_points(surface_1, surface_2)
