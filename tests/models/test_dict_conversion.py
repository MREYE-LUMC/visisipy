"""Tests for to_dict / from_dict conversion on eye-model classes."""

from __future__ import annotations

import json
import math

import pytest

from visisipy.models.base import EyeModel
from visisipy.models.geometry import (
    BiconicSurface,
    EyeGeometry,
    NoSurface,
    StandardSurface,
    Stop,
    Surface,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)
from visisipy.models.materials import (
    BennettRabbettsMaterials,
    EyeMaterials,
    GullstrandLeGrandAccommodatedMaterials,
    GullstrandLeGrandUnaccommodatedMaterials,
    MaterialModel,
    NavarroMaterials,
    NavarroMaterials458,
    NavarroMaterials543,
    NavarroMaterials589,
    NavarroMaterials633,
)


class TestMaterialModelDict:
    def test_to_dict(self):
        material = MaterialModel(refractive_index=1.5, abbe_number=50.0, partial_dispersion=0.06)
        result = material.to_dict()

        assert result == {"refractive_index": 1.5, "abbe_number": 50.0, "partial_dispersion": 0.06}

    def test_to_dict_defaults(self):
        material = MaterialModel(refractive_index=1.4)
        result = material.to_dict()

        assert result == {"refractive_index": 1.4, "abbe_number": 0.0, "partial_dispersion": 0.0}

    def test_from_dict(self):
        data = {"refractive_index": 1.5, "abbe_number": 50.0, "partial_dispersion": 0.06}
        material = MaterialModel.from_dict(data)

        assert material.refractive_index == 1.5
        assert material.abbe_number == 50.0
        assert material.partial_dispersion == 0.06

    def test_roundtrip(self):
        original = MaterialModel(refractive_index=1.376, abbe_number=55.0, partial_dispersion=0.01)
        reconstructed = MaterialModel.from_dict(original.to_dict())

        assert reconstructed == original


class TestEyeMaterialsDict:
    def test_to_dict_includes_type(self):
        materials = EyeMaterials(
            cornea=MaterialModel(1.376),
            aqueous=MaterialModel(1.336),
            lens=MaterialModel(1.42),
            vitreous=MaterialModel(1.336),
        )
        result = materials.to_dict()

        assert result["type"] == "EyeMaterials"

    def test_to_dict_serializes_material_models(self):
        cornea = MaterialModel(1.376, 55.0, 0.0)
        materials = EyeMaterials(
            cornea=cornea,
            aqueous=MaterialModel(1.336),
            lens=MaterialModel(1.42),
            vitreous=MaterialModel(1.336),
        )
        result = materials.to_dict()

        assert result["cornea"] == cornea.to_dict()

    def test_from_dict(self):
        data = {
            "type": "EyeMaterials",
            "cornea": {"refractive_index": 1.376, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "aqueous": {"refractive_index": 1.336, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "lens": {"refractive_index": 1.42, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "vitreous": {"refractive_index": 1.336, "abbe_number": 0.0, "partial_dispersion": 0.0},
        }
        materials = EyeMaterials.from_dict(data)

        assert isinstance(materials, EyeMaterials)
        assert materials.cornea.refractive_index == 1.376

    def test_from_dict_unknown_type_raises_valueerror(self):
        data = {
            "type": "UnknownMaterials",
            "cornea": {"refractive_index": 1.376, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "aqueous": {"refractive_index": 1.336, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "lens": {"refractive_index": 1.42, "abbe_number": 0.0, "partial_dispersion": 0.0},
            "vitreous": {"refractive_index": 1.336, "abbe_number": 0.0, "partial_dispersion": 0.0},
        }
        with pytest.raises(ValueError, match="Unknown materials type"):
            EyeMaterials.from_dict(data)

    def test_roundtrip(self):
        original = EyeMaterials(
            cornea=MaterialModel(1.376, 55.0, 0.0),
            aqueous=MaterialModel(1.336, 50.0, 0.0),
            lens=MaterialModel(1.42, 45.0, 0.0),
            vitreous=MaterialModel(1.336, 50.0, 0.0),
        )
        reconstructed = EyeMaterials.from_dict(original.to_dict())

        assert reconstructed == original


@pytest.mark.parametrize(
    "materials_cls",
    [
        NavarroMaterials,
        NavarroMaterials458,
        NavarroMaterials543,
        NavarroMaterials589,
        NavarroMaterials633,
        BennettRabbettsMaterials,
        GullstrandLeGrandUnaccommodatedMaterials,
        GullstrandLeGrandAccommodatedMaterials,
    ],
)
class TestEyeMaterialsSubclassDict:
    def test_to_dict_preserves_type(self, materials_cls):
        materials = materials_cls()
        result = materials.to_dict()

        assert result["type"] == materials_cls.__name__

    def test_roundtrip(self, materials_cls):
        original = materials_cls()
        reconstructed = EyeMaterials.from_dict(original.to_dict())

        assert type(reconstructed) is materials_cls
        assert reconstructed == original


class TestSurfaceDict:
    def test_standard_surface_to_dict(self):
        surface = StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55)
        result = surface.to_dict()

        assert result["type"] == "StandardSurface"
        assert result["radius"] == 7.72
        assert result["asphericity"] == -0.26
        assert result["thickness"] == 0.55
        assert result["semi_diameter"] is None
        assert result["is_stop"] is False

    def test_standard_surface_roundtrip(self):
        original = StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55)
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, StandardSurface)
        assert reconstructed == original

    def test_stop_to_dict(self):
        stop = Stop(semi_diameter=1.5, thickness=0.1)
        result = stop.to_dict()

        assert result["type"] == "Stop"
        assert result["semi_diameter"] == 1.5
        assert result["thickness"] == 0.1
        assert result["is_stop"] is True

    def test_stop_roundtrip(self):
        original = Stop(semi_diameter=1.5, thickness=0.1)
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, Stop)
        assert reconstructed == original

    def test_biconic_surface_roundtrip(self):
        original = BiconicSurface(radius=10.0, asphericity=-0.2, radius_x=8.0, asphericity_x=-0.1, thickness=0.5)
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, BiconicSurface)
        assert reconstructed == original

    def test_no_surface_to_dict(self):
        surface = NoSurface()
        result = surface.to_dict()

        assert result["type"] == "NoSurface"

    def test_no_surface_roundtrip(self):
        original = NoSurface()
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, NoSurface)
        assert reconstructed.thickness == original.thickness

    def test_zernike_sag_surface_roundtrip(self):
        original = ZernikeStandardSagSurface(
            zernike_coefficients={1: 0.1, 2: 0.2},
            maximum_term=4,
            extrapolate=False,
            norm_radius=50.0,
        )
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, ZernikeStandardSagSurface)
        assert reconstructed.zernike_coefficients == original.zernike_coefficients
        assert reconstructed.maximum_term == original.maximum_term
        assert reconstructed.extrapolate == original.extrapolate
        assert reconstructed.norm_radius == original.norm_radius

    def test_zernike_phase_surface_roundtrip(self):
        original = ZernikeStandardPhaseSurface(
            zernike_coefficients={1: 0.1, 3: 0.3},
            maximum_term=5,
            diffraction_order=2.0,
            norm_radius=75.0,
        )
        reconstructed = Surface.from_dict(original.to_dict())

        assert isinstance(reconstructed, ZernikeStandardPhaseSurface)
        assert reconstructed.zernike_coefficients == original.zernike_coefficients
        assert reconstructed.maximum_term == original.maximum_term
        assert reconstructed.diffraction_order == original.diffraction_order
        assert reconstructed.norm_radius == original.norm_radius

    def test_standard_surface_infinity_radius(self):
        original = StandardSurface()
        result = original.to_dict()

        assert math.isinf(result["radius"])
        reconstructed = Surface.from_dict(result)
        assert math.isinf(reconstructed.radius)

    def test_from_dict_unknown_type_raises_valueerror(self):
        data = {"type": "UnknownSurface", "thickness": 0.0}
        with pytest.raises(ValueError, match="Unknown surface type"):
            Surface.from_dict(data)


class TestEyeGeometryDict:
    def test_to_dict_includes_type(self, example_geometry):
        result = example_geometry.to_dict()

        assert result["type"] == "EyeGeometry"

    def test_to_dict_includes_all_surfaces(self, example_geometry):
        result = example_geometry.to_dict()

        assert "cornea_front" in result
        assert "cornea_back" in result
        assert "pupil" in result
        assert "lens_front" in result
        assert "lens_back" in result
        assert "retina" in result

    def test_to_dict_surfaces_are_dicts(self, example_geometry):
        result = example_geometry.to_dict()

        for surface_name in ("cornea_front", "cornea_back", "pupil", "lens_front", "lens_back", "retina"):
            assert isinstance(result[surface_name], dict)

    def test_from_dict(self, example_geometry):
        data = example_geometry.to_dict()
        reconstructed = EyeGeometry.from_dict(data)

        assert isinstance(reconstructed, EyeGeometry)

    def test_roundtrip_preserves_surface_types(self, example_geometry):
        data = example_geometry.to_dict()
        reconstructed = EyeGeometry.from_dict(data)

        assert type(reconstructed.cornea_front) is type(example_geometry.cornea_front)
        assert type(reconstructed.cornea_back) is type(example_geometry.cornea_back)
        assert type(reconstructed.pupil) is type(example_geometry.pupil)
        assert type(reconstructed.lens_front) is type(example_geometry.lens_front)
        assert type(reconstructed.lens_back) is type(example_geometry.lens_back)
        assert type(reconstructed.retina) is type(example_geometry.retina)

    def test_roundtrip_preserves_surface_values(self, example_geometry):
        data = example_geometry.to_dict()
        reconstructed = EyeGeometry.from_dict(data)

        assert reconstructed.cornea_front == example_geometry.cornea_front
        assert reconstructed.cornea_back == example_geometry.cornea_back
        assert reconstructed.pupil == example_geometry.pupil
        assert reconstructed.lens_front == example_geometry.lens_front
        assert reconstructed.lens_back == example_geometry.lens_back
        assert reconstructed.retina == example_geometry.retina

    def test_roundtrip_preserves_geometry_properties(self, example_geometry):
        data = example_geometry.to_dict()
        reconstructed = EyeGeometry.from_dict(data)

        assert reconstructed.axial_length == pytest.approx(example_geometry.axial_length)
        assert reconstructed.cornea_thickness == pytest.approx(example_geometry.cornea_thickness)
        assert reconstructed.anterior_chamber_depth == pytest.approx(example_geometry.anterior_chamber_depth)
        assert reconstructed.lens_thickness == pytest.approx(example_geometry.lens_thickness)
        assert reconstructed.vitreous_thickness == pytest.approx(example_geometry.vitreous_thickness)


class TestEyeModelDict:
    def test_to_dict_keys(self):
        model = EyeModel()
        result = model.to_dict()

        assert "geometry" in result
        assert "materials" in result

    def test_to_dict_geometry_is_dict(self):
        model = EyeModel()
        result = model.to_dict()

        assert isinstance(result["geometry"], dict)

    def test_to_dict_materials_is_dict(self):
        model = EyeModel()
        result = model.to_dict()

        assert isinstance(result["materials"], dict)

    def test_from_dict(self):
        model = EyeModel()
        data = model.to_dict()
        reconstructed = EyeModel.from_dict(data)

        assert isinstance(reconstructed, EyeModel)
        assert isinstance(reconstructed.geometry, EyeGeometry)
        assert isinstance(reconstructed.materials, EyeMaterials)

    def test_roundtrip_preserves_materials(self):
        model = EyeModel()
        data = model.to_dict()
        reconstructed = EyeModel.from_dict(data)

        assert type(reconstructed.materials) is type(model.materials)
        assert reconstructed.materials == model.materials

    def test_roundtrip_preserves_geometry_surfaces(self):
        model = EyeModel()
        data = model.to_dict()
        reconstructed = EyeModel.from_dict(data)

        geometry = model.geometry
        rec_geometry = reconstructed.geometry

        assert rec_geometry.cornea_front == geometry.cornea_front
        assert rec_geometry.cornea_back == geometry.cornea_back
        assert rec_geometry.pupil == geometry.pupil
        assert rec_geometry.lens_front == geometry.lens_front
        assert rec_geometry.lens_back == geometry.lens_back
        assert rec_geometry.retina == geometry.retina

    def test_to_json_includes_visisipy_version(self):
        model = EyeModel()
        json_data = model.to_json()
        parsed = json.loads(json_data)

        assert "visisipy_version" in parsed
        assert "geometry" in parsed
        assert "materials" in parsed

    def test_from_json(self):
        model = EyeModel()
        json_data = model.to_json()
        reconstructed = EyeModel.from_json(json_data)

        assert isinstance(reconstructed, EyeModel)
        assert reconstructed.materials == model.materials
        assert reconstructed.geometry.cornea_front == model.geometry.cornea_front

    def test_from_json_missing_visisipy_version_raises_valueerror(self):
        model = EyeModel()
        data = model.to_dict()
        json_data = json.dumps(data)

        with pytest.raises(
            ValueError,
            match="JSON data is missing required 'visisipy_version' key. Ensure the data was created with EyeModel.to_json().",
        ):
            EyeModel.from_json(json_data)

    def test_save_and_load_json(self, tmp_path):
        model = EyeModel()
        path = tmp_path / "eye_model.json"

        model.save_json(path)
        loaded = EyeModel.load_json(path)

        assert loaded.materials == model.materials
        assert loaded.geometry.cornea_front == model.geometry.cornea_front
        assert loaded.geometry.cornea_back == model.geometry.cornea_back
        assert loaded.geometry.pupil == model.geometry.pupil
        assert loaded.geometry.lens_front == model.geometry.lens_front
        assert loaded.geometry.lens_back == model.geometry.lens_back
        assert loaded.geometry.retina == model.geometry.retina
