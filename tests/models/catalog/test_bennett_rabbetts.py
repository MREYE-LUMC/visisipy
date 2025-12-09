from __future__ import annotations

import pytest

import visisipy
from visisipy.models.catalog.bennett_rabbetts import BennettRabbettsGeometry, surfaces_by_accommodation
from visisipy.models.geometry import StandardSurface
from visisipy.models.materials import BennettRabbettsMaterials


class TestBennettRabbettsGeometry:
    @pytest.mark.parametrize("accommodation", surfaces_by_accommodation.keys())
    def test_axial_length(self, accommodation):
        geometry = BennettRabbettsGeometry(accommodation)

        assert geometry.axial_length == pytest.approx(24.09)

    @pytest.mark.parametrize(
        "accommodation,expected_front_focal_length,expected_back_focal_length",
        [
            (0.0, -16.67, 22.27),
            (2.5, -15.91, 21.26),
            (5.0, -15.24, 20.36),
            (7.5, -14.62, 19.53),
            (10.0, -14.06, 18.79),
        ],
    )
    def test_focal_lengths(
        self, accommodation, expected_front_focal_length, expected_back_focal_length, optiland_backend
    ):
        model = visisipy.EyeModel(
            geometry=BennettRabbettsGeometry(accommodation),
            materials=BennettRabbettsMaterials(),
        )
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)

        front_focal_length = round(cardinal_points.focal_lengths.object, 2)
        back_focal_length = round(cardinal_points.focal_lengths.image, 2)

        assert front_focal_length == expected_front_focal_length
        assert back_focal_length == expected_back_focal_length

    @pytest.mark.parametrize(
        "accommodation,expected_front_focal_point,expected_back_focal_point",
        [
            (0.0, -15.16, 24.09),
            (2.5, -14.29, 23.21),
            (5.0, -13.53, 22.41),
            (7.5, -12.82, 21.68),
            (10.0, -12.19, 21.01),
        ],
    )
    def test_focal_points(self, accommodation, expected_front_focal_point, expected_back_focal_point, optiland_backend):
        model = visisipy.EyeModel(
            geometry=BennettRabbettsGeometry(accommodation),
            materials=BennettRabbettsMaterials(),
        )
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)

        front_focal_point = round(cardinal_points.focal_points.object, 2)
        back_focal_point = round(model.geometry.axial_length + cardinal_points.focal_points.image, 2)

        assert front_focal_point == expected_front_focal_point
        assert back_focal_point == expected_back_focal_point

    @pytest.mark.parametrize(
        "accommodation,expected_front_principal_point,expected_back_principal_point",
        [
            (0.0, 1.51, 1.82),
            (2.5, 1.62, 1.95),
            (5.0, 1.71, 2.05),
            (7.5, 1.80, 2.15),
            (10.0, 1.87, 2.23),
        ],
    )
    def test_principal_points(
        self, accommodation, expected_front_principal_point, expected_back_principal_point, optiland_backend
    ):
        model = visisipy.EyeModel(
            geometry=BennettRabbettsGeometry(accommodation),
            materials=BennettRabbettsMaterials(),
        )
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)

        front_principal_point = round(cardinal_points.principal_points.object, 2)
        back_principal_point = round(model.geometry.axial_length + cardinal_points.principal_points.image, 2)

        assert front_principal_point == expected_front_principal_point
        assert back_principal_point == expected_back_principal_point

    @pytest.mark.parametrize(
        "accommodation,expected_front_nodal_point,expected_back_nodal_point",
        [
            (0.0, 7.11, 7.42),
            (2.5, 6.97, 7.29),
            (5.0, 6.83, 7.17),
            (7.5, 6.71, 7.06),
            (10.0, 6.60, 6.95),
        ],
    )
    def test_nodal_points(self, accommodation, expected_front_nodal_point, expected_back_nodal_point, optiland_backend):
        model = visisipy.EyeModel(
            geometry=BennettRabbettsGeometry(accommodation),
            materials=BennettRabbettsMaterials(),
        )
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)

        front_nodal_point = round(cardinal_points.nodal_points.object, 2)
        back_nodal_point = round(model.geometry.axial_length + cardinal_points.nodal_points.image, 2)

        assert front_nodal_point == expected_front_nodal_point
        assert back_nodal_point == expected_back_nodal_point

    def test_custom_surface_does_not_overwrite_default_values(self):
        geometry = BennettRabbettsGeometry(
            0,
            lens_back=StandardSurface(radius=-6.5, asphericity=0, thickness=17),
        )

        assert geometry.lens_back.radius == -6.5
        assert geometry.lens_back.asphericity == 0
        assert geometry.lens_back.thickness == 17

        assert geometry.lens_back != surfaces_by_accommodation[0]["lens_back"]

    def test_invalid_accommodation_raises_valueerror(self):
        with pytest.raises(ValueError, match=r"Accommodation value 3\.0 not available\. Available values are:"):
            BennettRabbettsGeometry(3.0)
