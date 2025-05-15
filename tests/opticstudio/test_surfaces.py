from __future__ import annotations

import math
from contextlib import nullcontext as does_not_raise
from types import SimpleNamespace

import numpy as np
import pytest
from zospy.solvers import material_model as solve_material_model

from visisipy.models.geometry import (
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel
from visisipy.opticstudio.surfaces import (
    BaseOpticStudioZernikeSurface,
    OpticStudioNoSurface,
    OpticStudioSurface,
    OpticStudioSurfaceDataProperty,
    OpticStudioSurfaceProperty,
    OpticStudioZernikeStandardPhaseSurface,
    OpticStudioZernikeStandardSagSurface,
    make_surface,
)
from visisipy.wavefront import ZernikeCoefficients

pytestmark = [pytest.mark.needs_opticstudio]


@pytest.fixture
def surface(oss):
    surface = oss.LDE.InsertNewSurfaceAt(1)
    surface.Comment = "Test comment"

    return surface


@pytest.mark.parametrize("surface_class", [OpticStudioSurface, OpticStudioZernikeStandardSagSurface])
def test_surface_type_exists(zos, surface_class: OpticStudioSurface):
    assert hasattr(zos.ZOSAPI.Editors.LDE.SurfaceType, surface_class._TYPE)


class TestOpticStudioSurfaceProperty:
    MockSurface = SimpleNamespace(
        Comment="Test comment",
    )

    class MockOpticStudioSurface:
        surface = None

        def __init__(self, surface):
            self.surface = surface

        comment = OpticStudioSurfaceProperty("Comment")

    def test_init(self):
        prop = OpticStudioSurfaceProperty("Comment")
        assert prop.name == "Comment"

    def test_get(self):
        mock_surface = self.MockOpticStudioSurface(self.MockSurface)

        assert mock_surface.comment == "Test comment"

    def test_get_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        assert mock_surface.comment is None

    def test_set(self):
        surface = self.MockSurface
        mock_surface = self.MockOpticStudioSurface(surface)

        mock_surface.comment = "New comment"

        assert mock_surface.comment == "New comment"
        assert surface.Comment == "New comment"

    def test_set_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        with pytest.raises(AttributeError, match="Cannot set attribute .+ of non-built surface"):
            mock_surface.comment = "New comment"


class TestOpticStudioSurfaceDataProperty:
    MockSurface = SimpleNamespace(
        SurfaceData=SimpleNamespace(
            ExampleProperty="Example value",
        )
    )

    class MockOpticStudioSurface:
        surface = None

        def __init__(self, surface):
            self.surface = surface

        example_property = OpticStudioSurfaceDataProperty("ExampleProperty")

    def test_init(self):
        prop = OpticStudioSurfaceDataProperty("ExampleProperty")
        assert prop.name == "ExampleProperty"

    def test_get(self):
        mock_surface = self.MockOpticStudioSurface(self.MockSurface)

        assert mock_surface.example_property == "Example value"

    def test_get_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        assert mock_surface.example_property is None

    def test_set(self, monkeypatch):
        surface = self.MockSurface
        mock_surface = self.MockOpticStudioSurface(surface)

        monkeypatch.setattr(mock_surface, "example_property", "New value")

        assert mock_surface.example_property == "New value"
        assert surface.SurfaceData.ExampleProperty == "New value"

    def test_set_none(self):
        mock_surface = self.MockOpticStudioSurface(None)

        with pytest.raises(AttributeError, match="Cannot set attribute .+ of non-built surface"):
            mock_surface.example_property = "New value"


class TestOpticStudioSurface:
    @pytest.mark.parametrize("is_stop", [True, None])
    def test_build(self, oss, is_stop):
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
        surface_index = surface.build(oss, position=2)

        assert surface_index == 2
        assert surface._is_built is True

        assert surface.surface.SurfaceNumber == 2

        assert surface.comment == "Test comment"
        assert surface.radius == 1.0
        assert surface.thickness == 2.0
        assert surface.semi_diameter == 3.0
        assert surface.conic == 4.0
        assert surface.material == "TEST MATERIAL"  # OpticStudio capitalizes material names
        assert surface.is_stop is bool(is_stop)

    def test_is_stop_false_warns(self, oss):
        with pytest.warns(UserWarning, match="is_stop is set to False"):
            OpticStudioSurface(comment="Test", is_stop=False).build(oss, position=1)

    @pytest.mark.parametrize(
        "material_model",
        [
            MaterialModel(refractive_index=1.5, abbe_number=0, partial_dispersion=0),
            MaterialModel(refractive_index=1.5, abbe_number=50, partial_dispersion=0.67),
        ],
    )
    def test_set_material_model(self, oss, material_model):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=1)
        surface.material = material_model

        material_solvedata = surface.surface.MaterialCell.GetSolveData()._S_MaterialModel
        assert material_solvedata.IndexNd == material_model.refractive_index
        assert material_solvedata.AbbeVd == material_model.abbe_number
        assert material_solvedata.dPgF == material_model.partial_dispersion

    @pytest.mark.parametrize(
        "refractive_index,abbe_number,partial_dispersion",
        [(1.5, 0, 0), (1.5, 50, 0.67)],
    )
    def test_get_material_model(self, oss, refractive_index, abbe_number, partial_dispersion):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=1)
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

    def test_set_material_str(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=1)
        surface.material = "Test material"

        assert surface.surface.Material == "TEST MATERIAL"

    def test_get_material_str(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=1)
        surface.surface.Material = "Test material"

        assert surface.material == "TEST MATERIAL"

    def test_set_material_incorrect_type_raises_typeerror(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=1)

        with pytest.raises(TypeError, match="'material' must be MaterialModel or str"):
            surface.material = 5

    def test_relink_surface(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=2)

        assert surface.surface.SurfaceNumber == 2

        OpticStudioSurface(comment="New test").build(oss, position=1)

        assert surface.relink_surface(oss)
        assert surface.surface.SurfaceNumber == 3

    def test_relink_surface_changed_comment(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface.build(oss, position=2)
        surface._comment = "New test"

        assert not surface.relink_surface(oss)

    def test_set_surface_type(self, oss):
        surface = OpticStudioSurface(comment="Test")
        surface._TYPE = "ABCD"
        surface.build(oss, position=1)

        assert surface.surface.TypeName == "ABCD"

    def test_build_returns_index(self, oss):
        surface = OpticStudioSurface(comment="Test")

        assert surface.build(oss, position=1) == 1


class TestBaseOpticStudioZernikeSurface:
    class MockOpticStudioZernikeSurface(BaseOpticStudioZernikeSurface):
        """Very basic implementation of BaseOpticStudioZernikeSurface for testing purposes.

        This class is necessary because BaseOpticStudioZernikeSurface cannot be instantiated directly.
        """

        _TYPE = "ZernikeStandardSag"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def test_instantiation_raises_typeerror(self):
        with pytest.raises(TypeError, match="Only child classes of BaseOpticStudioZernikeSurface may be instantiated"):
            BaseOpticStudioZernikeSurface("Useless comment")

    def test_init_large_zernike_coefficient_raises_valueerror(self):
        with pytest.raises(ValueError, match="Zernike coefficients must be smaller than the maximum term 1"):
            self.MockOpticStudioZernikeSurface(
                comment="Test",
                number_of_terms=1,
                zernike_coefficients={2: math.pi},
            )

    @pytest.mark.parametrize("key", [0, -1, -math.pi])
    def test_init_negative_zernike_coefficient_raises_valueerror(self, key):
        with pytest.raises(ValueError, match="Zernike coefficients must be larger than 0"):
            self.MockOpticStudioZernikeSurface(
                comment="Test",
                number_of_terms=3,
                zernike_coefficients={key: 1.234},
            )

    @pytest.mark.parametrize(
        "n,value,maximum_term,expectation",
        [
            (0, 1.234, 1, pytest.raises(ValueError, match="Zernike coefficient must be larger than 0")),
            (1, 1.234, 2, does_not_raise()),
            (2, 1.234, 2, does_not_raise()),
            (
                2,
                1.234,
                1,
                pytest.raises(ValueError, match="Zernike coefficient must be smaller than the maximum term 1"),
            ),
        ],
    )
    def test_set_zernike_coefficient(self, oss, n: int, value: float, maximum_term: int, expectation):
        surface = self.MockOpticStudioZernikeSurface(
            comment="Test",
            number_of_terms=maximum_term,
        )

        surface.build(oss, position=1)

        with expectation:
            surface.set_zernike_coefficient(n, value)
            assert surface.surface.SurfaceData.GetNthZernikeCoefficient(n) == value

    @pytest.mark.parametrize(
        "n,coefficients,expectation",
        [
            (2, {1: 1.234, 2: 3.456}, does_not_raise()),
            (
                4,
                {1: 1.234, 2: 3.456},
                pytest.raises(ValueError, match="Zernike coefficient must be smaller than the maximum term 3"),
            ),
            (0, {1: 1.234, 2: 3.456}, pytest.raises(ValueError, match="Zernike coefficient must be larger than 0")),
        ],
    )
    def test_get_zernike_coefficient(self, oss, n, coefficients, expectation):
        surface = OpticStudioZernikeStandardSagSurface(
            comment="Test",
            number_of_terms=3,
            zernike_coefficients=coefficients,
        )

        surface.build(oss, position=1)

        with expectation:
            assert surface.get_zernike_coefficient(n) == coefficients[n]


class TestOpticStudioZernikeStandardSagSurface:
    @pytest.mark.parametrize(
        "maximum_term,coefficients,extrapolate,decenter_x,decenter_y",
        [
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 1, 0.0, 0.0),
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 0, -1.0, 1.0),
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 1, 1.0, -1.0),
            (231, {}, 0, 0.0, 0.0),
            (231, None, 1, 0.0, 0.0),
        ],
    )
    def test_build(self, oss, maximum_term, coefficients, extrapolate, decenter_x, decenter_y):
        surface = OpticStudioZernikeStandardSagSurface(
            comment="Test comment",
            number_of_terms=maximum_term,
            extrapolate=extrapolate,
            zernike_decenter_x=decenter_x,
            zernike_decenter_y=decenter_y,
            zernike_coefficients=coefficients,
        )

        assert surface._is_built is False
        surface_index = surface.build(oss, position=1)

        assert surface_index == 1
        assert surface._is_built is True
        assert str(surface.surface.Type) == "ZernikeStandardSag"

        assert surface.comment == "Test comment"
        assert surface.number_of_terms == maximum_term
        assert surface.extrapolate == extrapolate
        assert surface.zernike_decenter_x == decenter_x
        assert surface.zernike_decenter_y == decenter_y

        for n, value in surface._zernike_coefficients.items():
            assert surface.get_zernike_coefficient(n) == value

    @pytest.mark.parametrize("decenter_x", np.arange(-2, 3, dtype=float))
    def test_zernike_decenter_x(self, oss, decenter_x):
        surface = OpticStudioZernikeStandardSagSurface(
            comment="Test",
            zernike_decenter_x=decenter_x,
        )

        surface.build(oss, position=1)

        surface.zernike_decenter_x = decenter_x
        assert surface.zernike_decenter_x == decenter_x
        assert surface.surface.SurfaceData.ZernikeDecenter_X == decenter_x

    @pytest.mark.parametrize("decenter_y", np.arange(-2, 3, dtype=float))
    def test_zernike_decenter_y(self, oss, decenter_y):
        surface = OpticStudioZernikeStandardSagSurface(
            comment="Test",
            zernike_decenter_y=decenter_y,
        )

        surface.build(oss, position=1)

        surface.zernike_decenter_y = decenter_y
        assert surface.zernike_decenter_y == decenter_y
        assert surface.surface.SurfaceData.ZernikeDecenter_Y == decenter_y

    @pytest.mark.parametrize("extrapolate", [0, 1, True, False])
    def test_extrapolate(self, oss, extrapolate):
        surface = OpticStudioZernikeStandardSagSurface(
            comment="Test",
            extrapolate=extrapolate,
        )

        surface.build(oss, position=1)

        surface.extrapolate = extrapolate
        assert surface.extrapolate == extrapolate
        assert surface.surface.SurfaceData.Extrapolate == extrapolate


class TestOpticStudioZernikeStandardPhaseSurface:
    @pytest.mark.parametrize(
        "maximum_term,coefficients,extrapolate,diffract_order",
        [
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 1, 1),
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 0, 2.3),
            (3, {1: 1.0, 2: 2.0, 3: 3.0}, 1, 3.4),
            (231, {}, 0, 4.5),
            (231, None, 1, 5.6),
        ],
    )
    def test_build(self, oss, maximum_term, coefficients, extrapolate, diffract_order):
        surface = OpticStudioZernikeStandardPhaseSurface(
            comment="Test comment",
            number_of_terms=maximum_term,
            extrapolate=extrapolate,
            diffract_order=diffract_order,
            zernike_coefficients=coefficients,
        )

        assert surface._is_built is False
        surface_index = surface.build(oss, position=1)

        assert surface_index == 1
        assert surface._is_built is True
        assert str(surface.surface.Type) == "ZernikeStandardPhase"

        assert surface.comment == "Test comment"
        assert surface.number_of_terms == maximum_term
        assert surface.extrapolate == extrapolate
        assert surface.diffract_order == diffract_order

        for n, value in surface._zernike_coefficients.items():
            assert surface.get_zernike_coefficient(n) == value

    @pytest.mark.parametrize("diffract_order", np.arange(0, 4, dtype=float))
    def test_diffract_order(self, oss, diffract_order):
        surface = OpticStudioZernikeStandardPhaseSurface(
            comment="Test",
            diffract_order=diffract_order,
        )

        surface.build(oss, position=1)

        surface.diffract_order = diffract_order
        assert surface.diffract_order == diffract_order
        assert surface.surface.SurfaceData.DiffractOrder == diffract_order

    @pytest.mark.parametrize("extrapolate", [0, 1, True, False])
    def test_extrapolate(self, oss, extrapolate):
        surface = OpticStudioZernikeStandardPhaseSurface(
            comment="Test",
            extrapolate=extrapolate,
        )

        surface.build(oss, position=1)

        surface.extrapolate = extrapolate
        assert surface.extrapolate == extrapolate
        assert surface.surface.SurfaceData.Extrapolate == extrapolate


class TestNoSurface:
    def test_build(self, oss):
        surface = OpticStudioNoSurface()

        n_surfaces = oss.LDE.NumberOfSurfaces

        return_index = surface.build(oss, position=1)

        assert return_index == 0
        assert surface.surface is None
        assert n_surfaces == oss.LDE.NumberOfSurfaces


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
                MaterialModel(refractive_index=1.5, abbe_number=50, partial_dispersion=0.67),
            ),
        ],
    )
    def test_make_standard_surface(self, radius, thickness, semi_diameter, asphericity, material):
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
                MaterialModel(refractive_index=1.5, abbe_number=50, partial_dispersion=0.67),
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

    def test_make_zernike_standard_sag_surface(self):
        surface = ZernikeStandardSagSurface(
            radius=1.0,
            thickness=2.0,
            semi_diameter=0.5,
            asphericity=0.1,
            zernike_decenter_x=0.2,
            zernike_decenter_y=0.3,
            maximum_term=3,
            norm_radius=50,
            extrapolate=False,
            zernike_coefficients=ZernikeCoefficients({1: 1.0, 2: 2.0, 3: 3.0}),
        )

        opticstudio_surface = make_surface(surface, material="BK7")

        assert opticstudio_surface._radius == 1.0
        assert opticstudio_surface._thickness == 2.0
        assert opticstudio_surface._semi_diameter == 0.5
        assert opticstudio_surface._conic == 0.1
        assert opticstudio_surface._zernike_decenter_x == 0.2
        assert opticstudio_surface._zernike_decenter_y == 0.3
        assert opticstudio_surface._number_of_terms == 3
        assert opticstudio_surface._norm_radius == 50
        assert opticstudio_surface._extrapolate == 0
        assert opticstudio_surface._zernike_coefficients == {1: 1.0, 2: 2.0, 3: 3.0}
        assert opticstudio_surface._material == "BK7"

    def test_make_zernike_standard_phase_surface(self):
        surface = ZernikeStandardPhaseSurface(
            radius=1.0,
            thickness=2.0,
            semi_diameter=0.5,
            asphericity=0.1,
            diffraction_order=2,
            maximum_term=3,
            norm_radius=50,
            extrapolate=True,
            zernike_coefficients=ZernikeCoefficients({1: 1.0, 2: 2.0, 3: 3.0}),
        )

        opticstudio_surface = make_surface(surface, material="BK7")

        assert opticstudio_surface._radius == 1.0
        assert opticstudio_surface._thickness == 2.0
        assert opticstudio_surface._semi_diameter == 0.5
        assert opticstudio_surface._conic == 0.1
        assert opticstudio_surface._diffract_order == 2
        assert opticstudio_surface._number_of_terms == 3
        assert opticstudio_surface._norm_radius == 50
        assert opticstudio_surface._extrapolate == 1
        assert opticstudio_surface._zernike_coefficients == {1: 1.0, 2: 2.0, 3: 3.0}
        assert opticstudio_surface._material == "BK7"

    def test_make_no_surface(self):
        surface = NoSurface()
        opticstudio_surface = make_surface(surface)

        assert isinstance(opticstudio_surface, OpticStudioNoSurface)
