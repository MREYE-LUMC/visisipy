from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import optiland.materials
import pytest

from visisipy.models.geometry import NoSurface, StandardSurface, Stop, Surface
from visisipy.models.materials import MaterialModel
from visisipy.optiland.surfaces import (
    OptilandNoSurface,
    OptilandSurface,
    _built_only_property,
    make_surface,
)

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


def build_surface(optic: Optic, surface: OptilandSurface) -> int:
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

        assert surface._is_built is False
        assert surface.surface is None
        assert surface._optic is None
        assert surface._index is None

        surface_index = build_surface(optic, surface)

        assert surface_index == 1
        assert surface.surface is not None
        assert surface._is_built is True
        assert surface._optic == optic
        assert surface._index == 1

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

    def test_make_no_surface(self):
        surface = NoSurface()
        opticstudio_surface = make_surface(surface)

        assert isinstance(opticstudio_surface, OptilandNoSurface)
