from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from visisipy.analysis import cardinal_points
from visisipy.models import EyeModel
from visisipy.models.materials import NavarroMaterials543

if TYPE_CHECKING:
    from visisipy.analysis.cardinal_points import CardinalPointsResult

ABSOLUTE_TOLERANCE = 1e-6
EXPECTED_CARDINAL_POINTS_RESULT = {
    "focal_length_object": -16.467904,
    "focal_length_image": 22.029115,
    "focal_point_object": -14.885414,
    "focal_point_image": 0.000014,
    "principal_point_object": 1.582490,
    "principal_point_image": -22.029102,
    "anti_principal_point_object": -31.353319,
    "anti_principal_point_image": 22.029129,
    "nodal_point_object": 7.143701,
    "nodal_point_image": -16.467890,
    "anti_nodal_point_object": -36.914530,
    "anti_nodal_point_image": 16.467918,
}


@pytest.fixture
def cardinal_points_result(configure_backend) -> CardinalPointsResult:
    model = EyeModel(materials=NavarroMaterials543())

    return cardinal_points(model=model, backend=configure_backend)


class TestCardinalPoints:
    def test_focal_lengths(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.focal_lengths.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["focal_length_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.focal_lengths.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["focal_length_image"], abs=ABSOLUTE_TOLERANCE
        )

    def test_focal_points(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.focal_points.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["focal_point_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.focal_points.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["focal_point_image"], abs=ABSOLUTE_TOLERANCE
        )

    def test_principal_points(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.principal_points.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["principal_point_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.principal_points.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["principal_point_image"], abs=ABSOLUTE_TOLERANCE
        )

    def test_anti_principal_points(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.anti_principal_points.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["anti_principal_point_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.anti_principal_points.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["anti_principal_point_image"], abs=ABSOLUTE_TOLERANCE
        )

    def test_nodal_points(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.nodal_points.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["nodal_point_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.nodal_points.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["nodal_point_image"], abs=ABSOLUTE_TOLERANCE
        )

    def test_anti_nodal_points(self, cardinal_points_result: CardinalPointsResult):
        assert cardinal_points_result.anti_nodal_points.object == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["anti_nodal_point_object"], abs=ABSOLUTE_TOLERANCE
        )
        assert cardinal_points_result.anti_nodal_points.image == pytest.approx(
            EXPECTED_CARDINAL_POINTS_RESULT["anti_nodal_point_image"], abs=ABSOLUTE_TOLERANCE
        )
