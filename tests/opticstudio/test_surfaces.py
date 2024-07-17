import pytest

from visisipy.models.materials import MaterialModel
from visisipy.opticstudio.surfaces import (
    OpticStudioSurface,
    OpticStudioSurfaceProperty,
    make_surface,
)
from visisipy.models.geometry import StandardSurface, Stop, Surface

from zospy.solvers import material_model as solve_material_model


@pytest.fixture
def surface(new_oss):
    surface = new_oss.LDE.InsertNewSurfaceAt(1)
    surface.Comment = "Test comment"

    return surface


class TestOpticStudioSurfaceProperty:
    class MockOpticStudioSurface:
        surface = None

        def __init__(self, surface):
            self.surface = surface

        comment = OpticStudioSurfaceProperty("Comment")

    def test_init(self):
        prop = OpticStudioSurfaceProperty("Comment")
        assert prop.name == "Comment"

    def test_get(self, surface):
        mock_surface = self.MockOpticStudioSurface(surface)

        assert mock_surface.comment == "Test comment"

    def test_get_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        assert mock_surface.comment is None

    def test_set(self, surface):
        mock_surface = self.MockOpticStudioSurface(surface)

        mock_surface.comment = "New comment"

        assert mock_surface.comment == "New comment"
        assert surface.Comment == "New comment"

    def test_set_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        with pytest.raises(
            AttributeError, match="Cannot set attribute .+ of non-built surface"
        ):
            mock_surface.comment = "New comment"


class TestOpticStudioSurface:
    @pytest.mark.parametrize("is_stop", [True, None])
    def test_build(self, new_oss, is_stop):
        surface = OpticStudioSurface(
            comment="Test comment",
            radius=1.0,
            thickness=2.0,
            semi_diameter=3.0,
            conic=4.0,
            material="Test material",
            is_stop=is_stop,
        )

        assert surface._is_built is False
        surface.build(new_oss, position=2)
        assert surface._is_built is True

        assert surface.surface.SurfaceNumber == 2

        assert surface.comment == "Test comment"
        assert surface.radius == 1.0
        assert surface.thickness == 2.0
        assert surface.semi_diameter == 3.0
        assert surface.conic == 4.0
        assert (
            surface.material == "TEST MATERIAL"
        )  # OpticStudio capitalizes material names
        assert surface.is_stop is bool(is_stop)

    def test_is_stop_false_warns(self, new_oss):
        with pytest.warns(UserWarning, match="is_stop is set to False"):
            OpticStudioSurface(comment="Test", is_stop=False).build(new_oss, position=1)

    @pytest.mark.parametrize(
        "material_model",
        [
            MaterialModel(refractive_index=1.5, abbe_number=0, partial_dispersion=0),
            MaterialModel(
                refractive_index=1.5, abbe_number=50, partial_dispersion=0.67
            ),
        ],
    )
    def test_set_material_model(self, new_oss, material_model):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=1)
        surface.material = material_model

        material_solvedata = (
            surface.surface.MaterialCell.GetSolveData()._S_MaterialModel
        )
        assert material_solvedata.IndexNd == material_model.refractive_index
        assert material_solvedata.AbbeVd == material_model.abbe_number
        assert material_solvedata.dPgF == material_model.partial_dispersion

    @pytest.mark.parametrize(
        "refractive_index,abbe_number,partial_dispersion",
        [(1.5, 0, 0), (1.5, 50, 0.67)],
    )
    def test_get_material_model(
        self, new_oss, refractive_index, abbe_number, partial_dispersion
    ):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=1)
        solve_material_model(
            surface.surface.MaterialCell,
            refractive_index=refractive_index,
            abbe_number=abbe_number,
            partial_dispersion=partial_dispersion,
        )

        assert surface.material == MaterialModel(
            refractive_index=refractive_index,
            abbe_number=abbe_number,
            partial_dispersion=partial_dispersion,
        )

    def test_get_material_before_build(self):
        surface = OpticStudioSurface(comment="Test", material="Test material")

        assert surface.material is None

    def test_set_material_str(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=1)
        surface.material = "Test material"

        assert surface.surface.Material == "TEST MATERIAL"

    def test_get_material_str(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=1)
        surface.surface.Material = "Test material"

        assert surface.material == "TEST MATERIAL"

    def test_set_material_incorrect_type_raises_valueerror(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=1)

        with pytest.raises(ValueError, match="'material' must be MaterialModel or str"):
            surface.material = 5

    def test_relink_surface(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=2)

        assert surface.surface.SurfaceNumber == 2

        OpticStudioSurface(comment="New test").build(new_oss, position=1)

        assert surface.relink_surface(new_oss)
        assert surface.surface.SurfaceNumber == 3

    def test_relink_surface_changed_comment(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(new_oss, position=2)
        surface._comment = "New test"

        assert not surface.relink_surface(new_oss)

    def test_set_surface_type(self, new_oss):
        surface = OpticStudioSurface(comment="Test")
        surface._TYPE = "ABCD"
        surface.build(new_oss, position=1)

        assert surface.surface.TypeName == "ABCD"


class TestMakeSurface:
    def test_make_surface(self):
        surface = Surface(thickness=1)

        opticstudio_surface = make_surface(surface, material="BK7")

        assert opticstudio_surface._thickness == 1
        assert opticstudio_surface._material == "BK7"

    @pytest.mark.parametrize(
        "radius,thickness,semi_diameter,asphericity,material",
        [
            (1, 2, 3, 4, "BK7"),
            (
                5,
                6,
                7,
                8,
                MaterialModel(
                    refractive_index=1.5, abbe_number=50, partial_dispersion=0.67
                ),
            ),
        ],
    )
    def test_make_standard_surface(
        self, radius, thickness, semi_diameter, asphericity, material
    ):
        surface = StandardSurface(
            radius=radius,
            thickness=thickness,
            semi_diameter=semi_diameter,
            asphericity=asphericity,
        )

        opticstudio_surface = make_surface(surface, material)

        assert opticstudio_surface._radius == radius
        assert opticstudio_surface._thickness == thickness
        assert opticstudio_surface._semi_diameter == semi_diameter
        assert opticstudio_surface._conic == asphericity
        assert opticstudio_surface._material == material

    @pytest.mark.parametrize(
        "thickness,semi_diameter,material",
        [
            (1, 2, "BK7"),
            (
                3,
                4,
                MaterialModel(
                    refractive_index=1.5, abbe_number=50, partial_dispersion=0.67
                ),
            ),
        ],
    )
    def test_make_stop_surface(self, thickness, semi_diameter, material):
        surface = Stop(thickness=thickness, semi_diameter=semi_diameter)

        opticstudio_surface = make_surface(surface, material)

        assert opticstudio_surface._thickness == thickness
        assert opticstudio_surface._semi_diameter == semi_diameter
        assert opticstudio_surface._material == material
        assert opticstudio_surface._is_stop is True
