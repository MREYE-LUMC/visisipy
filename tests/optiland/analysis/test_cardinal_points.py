from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.analysis.cardinal_points import CardinalPoints
from visisipy.models import EyeModel
from visisipy.optiland.analysis.cardinal_points import OptilandCardinalPointsResult


class TestCardinalPointsAnalysis:
    @pytest.mark.parametrize(
        "surface_1,surface_2,expectation",
        [
            (None, None, does_not_raise()),
            (1, 6, does_not_raise()),
            (2, 4, pytest.raises(ValueError, match="Optiland only supports calculating cardinal points for the entire system")),
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


class TestOptilandCardinalPointsResult:
    def test_init(self):
        result = OptilandCardinalPointsResult(
            focal_lengths=CardinalPoints(0, 0),
            focal_points=CardinalPoints(0, 0),
            principal_points=CardinalPoints(0, 0),
            nodal_points=CardinalPoints(0, 0),
        )

        assert result.focal_lengths == CardinalPoints(0, 0)
        assert result.focal_points  == CardinalPoints(0, 0)
        assert result.principal_points == CardinalPoints(0, 0)
        assert result.nodal_points == CardinalPoints(0, 0)
        assert result.anti_principal_points is NotImplemented
        assert result.anti_nodal_points is NotImplemented

    @pytest.mark.parametrize("property_", ["anti_principal_points", "anti_nodal_points"])
    def test_cannot_init_unsupported_properties(self, property_):
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            OptilandCardinalPointsResult(
                focal_lengths=CardinalPoints(0, 0),
                focal_points=CardinalPoints(0, 0),
                principal_points=CardinalPoints(0, 0),
                nodal_points=CardinalPoints(0, 0),
                **{property_: CardinalPoints(0, 0)},
            )
