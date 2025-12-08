from __future__ import annotations

import pytest

import visisipy
from visisipy.models.materials import GullstrandLeGrandAccommodatedMaterials, GullstrandLeGrandUnaccommodatedMaterials
from visisipy.models.zoo.gullstrand import Accommodation, GullstrandLeGrandGeometry


def make_gullstrand_model(accommodation: Accommodation) -> visisipy.EyeModel:
    geometry = GullstrandLeGrandGeometry(accommodation=accommodation)
    materials = (
        GullstrandLeGrandAccommodatedMaterials()
        if accommodation == "accommodated"
        else GullstrandLeGrandUnaccommodatedMaterials()
    )
    return visisipy.EyeModel(geometry=geometry, materials=materials)


class TestGullstrandLeGrandGeometry:
    @pytest.mark.parametrize(
        "accommodation,cornea_back,lens_front,lens_back,retina",
        [
            ("unaccommodated", 0.55, 3.6, 7.6, 24.19655),
            ("accommodated", 0.55, 3.2, 7.7, 24.19655),
        ],
    )
    def test_lengths(self, accommodation, cornea_back, lens_front, lens_back, retina):
        geometry = GullstrandLeGrandGeometry(accommodation=accommodation)

        assert geometry.cornea_thickness == cornea_back
        assert geometry.cornea_thickness + geometry.anterior_chamber_depth == pytest.approx(lens_front)
        assert geometry.cornea_thickness + geometry.anterior_chamber_depth + geometry.lens_thickness == pytest.approx(
            lens_back
        )
        assert geometry.axial_length == pytest.approx(retina)

    @pytest.mark.parametrize(
        "accommodation,expected_front_focal_length,expected_back_focal_length",
        [
            ("unaccommodated", -16.68, 22.29),
            ("accommodated", -14.78, 19.74),
        ],
    )
    def test_focal_length(
        self, accommodation, expected_front_focal_length, expected_back_focal_length, optiland_backend
    ):
        model = make_gullstrand_model(accommodation)
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)
        front_focal_length = round(cardinal_points.focal_lengths.object, 2)
        back_focal_length = round(cardinal_points.focal_lengths.image, 2)

        assert front_focal_length == pytest.approx(expected_front_focal_length)
        assert back_focal_length == pytest.approx(expected_back_focal_length)

    @pytest.mark.parametrize(
        "accommodation,expected_front_principal_point,expected_back_principal_point",
        [
            ("unaccommodated", 1.59, 1.91),
            ("accommodated", 1.82, 2.19),
        ],
    )
    def test_principal_points(
        self, accommodation, expected_front_principal_point, expected_back_principal_point, optiland_backend
    ):
        model = make_gullstrand_model(accommodation)
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)
        front_principal_point = round(cardinal_points.principal_points.object, 2)
        back_principal_point = round(model.geometry.axial_length + cardinal_points.principal_points.image, 2)

        assert front_principal_point == pytest.approx(expected_front_principal_point)
        assert back_principal_point == pytest.approx(expected_back_principal_point)

    @pytest.mark.parametrize(
        "accommodation,expected_front_nodal_point,expected_back_nodal_point",
        [
            ("unaccommodated", 7.20, 7.51),
            ("accommodated", 6.78, 7.16),
        ],
    )
    def test_nodal_points(self, accommodation, expected_front_nodal_point, expected_back_nodal_point, optiland_backend):
        model = make_gullstrand_model(accommodation)
        optiland_backend.build_model(model)

        cardinal_points = visisipy.analysis.cardinal_points(model, backend=optiland_backend)
        front_nodal_point = round(cardinal_points.nodal_points.object, 2)
        back_nodal_point = round(model.geometry.axial_length + cardinal_points.nodal_points.image, 2)

        assert front_nodal_point == pytest.approx(expected_front_nodal_point)
        assert back_nodal_point == pytest.approx(expected_back_nodal_point)

    def test_invalid_accommodation_raises_valueerror(self):
        with pytest.raises(ValueError, match="accommodation must be 'accommodated' or 'unaccommodated'"):
            GullstrandLeGrandGeometry(accommodation="invalid")
