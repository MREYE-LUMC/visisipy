from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import optiland.materials
import pytest

from visisipy.models.geometry import (
    BiconicSurface,
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import MaterialModel
from visisipy.optiland.surfaces import (
    BaseOptilandSurface,
    OptilandBiconicSurface,
    OptilandNoSurface,
    OptilandSurface,
    OptilandZernikeStandardSagSurface,
    _built_only_property,
    make_surface,
)
from visisipy.wavefront import ZernikeCoefficients

if TYPE_CHECKING:
    import optiland.surfaces  # noqa: TC004
    from optiland.optic import Optic


@pytest.fixture
def simple_system(optic: Optic) -> Optic:
    """Simple system with an object and image surface."""
    optic.add_surface(
        index=0,
        thickness=float("inf"),
        comment="Object",
    )

    optic.add_surface(
        index=1,
        radius=float("inf"),
        thickness=0,
        conic=0.0,
        comment="Image",
    )

    return optic


@pytest.fixture
def surface(simple_system: Optic) -> optiland.surfaces.Surface:
    """Fixture to create a surface for testing."""
    simple_system.add_surface(
        index=1,
        radius=1.0,
        thickness=2.0,
        conic=0.0,
        material="BK7",
        is_stop=True,
        comment="Test surface",
    )

    return simple_system.surface_group.surfaces[1]


class TestBuiltOnlyProperty:
    MockSurface = SimpleNamespace(
        comment="Test surface",
    )

    class MockOptilandSurface:
        surface: SimpleNamespace | None = None

        def __init__(self, surface) -> None:
            self.surface = surface
            self._is_built = False

        @_built_only_property
        def comment(self) -> str:
            return self.surface.comment  # type: ignore

        @comment.setter
        def comment(self, value: str) -> None:
            self.surface.comment = value  # type: ignore

        def build(self) -> None:
            self._is_built = True

    def test_get_built(self):
        """Test getting a built property."""
        mock_surface = self.MockOptilandSurface(self.MockSurface)
        mock_surface.build()

        assert mock_surface.comment == "Test surface"

    def test_get_not_built(self):
        """Test getting a property before it is built."""
        mock_surface = self.MockOptilandSurface(self.MockSurface)

        assert mock_surface.comment is None

    def test_set_built(self):
        """Test setting a built property."""
        mock_surface = self.MockOptilandSurface(self.MockSurface)
        mock_surface.build()

        mock_surface.comment = "New comment"

        assert mock_surface.surface.comment == "New comment"  # pyright: ignore [reportOptionalMemberAccess]

    def test_set_not_built(self):
        """Test setting a property before it is built."""
        mock_surface = self.MockOptilandSurface(self.MockSurface)

        with pytest.raises(AttributeError, match="Cannot set attribute of non-built surface"):
            mock_surface.comment = "New comment"


def build_surface(optic: Optic, surface: BaseOptilandSurface) -> int:
    """Build a surface with `surface_settings` to the optic system, between an object and image surface."""
    optic.add_surface(
        index=0,
        thickness=float("inf"),
        comment="Object",
    )

    index = surface.build(optic, position=1)

    optic.add_surface(
        index=2,
        radius=float("inf"),
        thickness=0,
        conic=0.0,
        comment="Image",
    )

    return index


def assert_pre_build(surface: BaseOptilandSurface) -> None:
    """Assert that the surface is in a pre-build state."""
    assert surface._is_built is False
    assert surface.surface is None
    assert surface._optic is None
    assert surface._index is None


def assert_post_build(surface: BaseOptilandSurface, optic: Optic, index: int) -> None:
    """Assert that the surface is in a post-build state."""
    assert surface._is_built is True
    assert surface.surface is not None
    assert surface._optic == optic
    assert surface._index == index


class TestOptilandSurface:
    @pytest.mark.parametrize("is_stop", [True, False, None])
    def test_build(self, optic: Optic, is_stop: bool):
        surface = OptilandSurface(
            comment="Test surface",
            radius=1.0,
            thickness=2.0,
            semi_diameter=1.0,
            conic=0.0,
            material="BK7",
            is_stop=is_stop,
        )

        assert_pre_build(surface)

        surface_index = build_surface(optic, surface)

        assert_post_build(surface, optic, surface_index)

        assert surface.comment == "Test surface"
        assert surface.radius == 1.0
        assert surface.thickness == 2.0
        assert surface.semi_diameter == 1.0
        assert surface.conic == 0.0
        assert surface.material == "BK7"
        assert surface.is_stop == bool(is_stop)

    def test_build_ideal_material(self, optic: Optic):
        material = MaterialModel(refractive_index=1.5, abbe_number=0.0, partial_dispersion=0.0)
        surface = OptilandSurface(
            comment="Test surface",
            radius=1.0,
            thickness=2.0,
            semi_diameter=1.0,
            conic=0.0,
            material=material,
            is_stop=True,
        )

        surface_index = build_surface(optic, surface)

        assert surface_index == 1
        assert surface.material == material
        assert isinstance(surface.surface.material_post, optiland.materials.IdealMaterial)
        assert surface.surface.material_post.index == material.refractive_index
        assert surface.surface.material_post.k(0.543) == 0

    def test_build_abbe_material(self, optic: Optic):
        material = MaterialModel(refractive_index=1.5, abbe_number=50.0, partial_dispersion=0.0)
        surface = OptilandSurface(
            comment="Test surface",
            radius=1.0,
            thickness=2.0,
            semi_diameter=1.0,
            conic=0.0,
            material=material,
            is_stop=True,
        )

        surface_index = build_surface(optic, surface)

        assert surface_index == 1
        assert surface.material == material
        assert isinstance(surface.surface.material_post, optiland.materials.AbbeMaterial)
        assert surface.surface.material_post.index == material.refractive_index
        assert surface.surface.material_post.abbe == material.abbe_number

    def test_build_str_material(self, optic: Optic):
        material = "BK7"
        surface = OptilandSurface(
            comment="Test surface",
            radius=1.0,
            thickness=2.0,
            semi_diameter=1.0,
            conic=0.0,
            material=material,
            is_stop=True,
        )

        surface_index = build_surface(optic, surface)

        assert surface_index == 1
        assert surface.material == material
        assert isinstance(surface.surface.material_post, optiland.materials.Material)
        assert surface.surface.material_post.name == material

    def test_set_material_incorrect_type_raises_typeerror(self, optic: Optic):
        surface = OptilandSurface(comment="Test surface", material=123)

        with pytest.raises(TypeError, match="'material' must be MaterialModel or str"):
            build_surface(optic, surface)


class TestOptilandBiconicSurface:
    @pytest.mark.parametrize("is_stop", [True, False, None])
    def test_build(self, optic: Optic, is_stop: bool):
        surface = OptilandBiconicSurface(
            comment="Test surface",
            radius=1.0,
            radius_x=2.0,
            thickness=3.0,
            semi_diameter=4.0,
            conic=0.5,
            conic_x=0.3,
            material="BK7",
            is_stop=is_stop,
        )

        assert_pre_build(surface)

        surface_index = build_surface(optic, surface)

        assert_post_build(surface, optic, surface_index)

        assert surface.comment == "Test surface"
        assert surface.radius == 1.0
        assert surface.radius_x == 2.0
        assert surface.thickness == 3.0
        assert surface.semi_diameter == 4.0
        assert surface.conic == 0.5
        assert surface.conic_x == 0.3
        assert surface.material == "BK7"
        assert surface.is_stop == bool(is_stop)

    def test_set_biconic_radii(self, optic: Optic):
        surface = OptilandBiconicSurface(
            comment="Test surface",
            radius=1.0,
            radius_x=2.0,
        )
        build_surface(optic, surface)

        assert surface.radius == 1.0
        assert surface.surface.geometry.Ry == 1.0
        assert surface.radius_x == 2.0
        assert surface.surface.geometry.Rx == 2.0

    def test_set_biconic_conics(self, optic: Optic):
        surface = OptilandBiconicSurface(
            comment="Test surface",
            conic=0.5,
            conic_x=0.3,
        )
        build_surface(optic, surface)

        assert surface.conic == 0.5
        assert surface.surface.geometry.ky == 0.5
        assert surface.conic_x == 0.3
        assert surface.surface.geometry.kx == 0.3


class TestOptilandZernikeSurface:
    @pytest.mark.parametrize("is_stop", [True, False, None])
    def test_build(self, optic: Optic, is_stop: bool):
        surface = OptilandZernikeStandardSagSurface(
            comment="Test Zernike surface",
            radius=1.0,
            thickness=2.0,
            semi_diameter=1.0,
            number_of_terms=5,
            norm_radius=1.0,
            zernike_coefficients=ZernikeCoefficients({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}),
            material="BK7",
            is_stop=is_stop,
        )

        assert_pre_build(surface)

        surface_index = build_surface(optic, surface)

        assert_post_build(surface, optic, surface_index)

        assert surface.comment == "Test Zernike surface"
        assert surface.radius == 1.0
        assert surface.thickness == 2.0
        assert surface.semi_diameter == 1.0
        assert surface.norm_radius == 1.0
        assert surface.coefficients == ZernikeCoefficients({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5})
        assert surface.material == "BK7"
        assert surface.is_stop == bool(is_stop)

    @pytest.mark.parametrize(
        "coefficients,number_of_terms,expectation",
        [
            (None, 0, does_not_raise()),
            (None, 5, does_not_raise()),
            ({1: 0.1, 4: 0.2}, 5, does_not_raise()),
            (ZernikeCoefficients({1: 0.1, 4: 0.2}), 5, does_not_raise()),
            (
                {1: 0.1, 6: 0.2},
                5,
                pytest.raises(ValueError, match="Zernike coefficients must be less than or equal to the maximum term"),
            ),
            (
                ZernikeCoefficients({1: 0.1, 6: 0.2}),
                5,
                pytest.raises(ValueError, match="Zernike coefficients must be less than or equal to the maximum term"),
            ),
            ({-1: 0.1}, 0, pytest.raises(ValueError, match="Zernike coefficients must be positive integers")),
            ({0: 0.1}, 0, pytest.raises(ValueError, match="Zernike coefficients must be positive integers")),
        ],
    )
    def test_set_coefficients(
        self,
        optic: Optic,
        coefficients: ZernikeCoefficients | dict[int, float] | None,
        number_of_terms: int,
        expectation,
    ):
        with expectation:
            surface = OptilandZernikeStandardSagSurface(
                comment="Test Zernike surface",
                number_of_terms=number_of_terms,
                zernike_coefficients=coefficients,
            )

            build_surface(optic, surface)

            if coefficients is None:
                assert all(c == 0.0 for c in surface.coefficients.values())
                assert len(surface.coefficients) == number_of_terms
                assert max(surface.coefficients.keys(), default=0) == number_of_terms
            else:
                assert all(
                    v == coefficients[c] if c in coefficients else v == 0 for c, v in surface.coefficients.items()
                )

    def test_get_norm_radius(self, optic: Optic):
        surface = OptilandZernikeStandardSagSurface(
            comment="Test Zernike surface",
            norm_radius=2.0,
            number_of_terms=5,
            zernike_coefficients=ZernikeCoefficients({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}),
        )

        build_surface(optic, surface)

        assert surface.norm_radius == 2.0
        assert surface.surface.geometry.norm_radius == 2.0  # type: ignore

    def test_get_coefficients(self, optic: Optic):
        coefficients = ZernikeCoefficients({1: 0.1, 2: 0.2, 3: 0.3})
        surface = OptilandZernikeStandardSagSurface(
            comment="Test Zernike surface",
            number_of_terms=5,
            zernike_coefficients=coefficients,
        )

        build_surface(optic, surface)

        assert all(surface.coefficients[c] == v for c, v in enumerate(surface.surface.geometry.coefficients, start=1))  # type: ignore


class TestNoSurface:
    def test_build(self, optic):
        surface = OptilandNoSurface()

        n_surfaces = optic.surface_group.num_surfaces

        return_index = surface.build(optic, position=1)

        assert return_index == 0
        assert surface.surface is None
        assert n_surfaces == optic.surface_group.num_surfaces


class TestMakeSurface:
    def test_make_surface(self):
        surface = Surface(thickness=1)

        optiland_surface = make_surface(surface, material="BK7")

        assert optiland_surface._thickness == 1
        assert optiland_surface._material == "BK7"

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

        optiland_surface = make_surface(surface, material)

        assert optiland_surface._radius == radius
        assert optiland_surface._thickness == thickness
        assert optiland_surface._semi_diameter == semi_diameter
        assert optiland_surface._conic == asphericity
        assert optiland_surface._material == material

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

        optiland_surface = make_surface(surface, material)

        assert optiland_surface._thickness == thickness
        assert optiland_surface._semi_diameter == semi_diameter
        assert optiland_surface._material == material
        assert optiland_surface._is_stop is True

    def test_make_biconic_surface(self):
        surface = BiconicSurface(radius=1, radius_x=2, thickness=2, semi_diameter=3, asphericity=0.5, asphericity_x=0.3)

        optiland_surface = make_surface(surface, material="BK7")

        assert isinstance(optiland_surface, OptilandBiconicSurface)
        assert optiland_surface._radius == 1
        assert optiland_surface._radius_x == 2
        assert optiland_surface._thickness == 2
        assert optiland_surface._semi_diameter == 3
        assert optiland_surface._conic == 0.5
        assert optiland_surface._conic_x == 0.3
        assert optiland_surface._material == "BK7"

    @pytest.mark.parametrize(
        "norm_radius,maximum_term,coefficients",
        [
            (2.0, 5, ZernikeCoefficients()),
            (1.5, 10, {1: 0.1, 2: 0.2, 3: 0.3}),
            (2.5, 15, ZernikeCoefficients({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})),
        ],
    )
    def test_make_zernike_standard_sag_surface(self, norm_radius, maximum_term, coefficients):
        surface = ZernikeStandardSagSurface(
            zernike_coefficients=coefficients,
            maximum_term=maximum_term,
            norm_radius=norm_radius,
        )

        optiland_surface = make_surface(surface, material="BK7")

        assert optiland_surface._zernike_coefficients == coefficients  # type: ignore

        assert optiland_surface._number_of_terms == maximum_term  # type: ignore
        assert optiland_surface._norm_radius == norm_radius  # type: ignore

    @pytest.mark.parametrize(
        "parameters,warning",
        [
            ({"extrapolate": True}, "Zernike surface extrapolation is not supported in Optiland."),
            ({"zernike_decenter_x": 1.0}, "Zernike surface decentering is not supported in Optiland."),
            ({"zernike_decenter_y": -1.0}, "Zernike surface decentering is not supported in Optiland."),
            (
                {"zernike_decenter_x": 1.0, "zernike_decenter_y": -1.0},
                "Zernike surface decentering is not supported in Optiland.",
            ),
        ],
    )
    def test_make_zernike_standard_sag_surface_unsupported_parameters(self, parameters: dict[str, Any], warning: str):
        surface = ZernikeStandardSagSurface(**parameters)

        with pytest.warns(UserWarning, match=warning):
            optiland_surface = make_surface(surface, material="BK7")

        assert isinstance(optiland_surface, OptilandZernikeStandardSagSurface)

    def test_make_no_surface(self):
        surface = NoSurface()
        opticstudio_surface = make_surface(surface)

        assert isinstance(opticstudio_surface, OptilandNoSurface)
