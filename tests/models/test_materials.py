from __future__ import annotations

from visisipy.models.materials import EyeMaterials, MaterialModel, NavarroMaterials


class TestMaterialModel:
    def test_initialization(self):
        material = MaterialModel(1.5, 1, 0.06)
        assert material.refractive_index == 1.5
        assert material.abbe_number == 1
        assert material.partial_dispersion == 0.06


class TestEyeMaterials:
    def test_initialization(self):
        cornea = MaterialModel(1.5, 50, 0.06)
        aqueous = MaterialModel(1.4, 45, 0.05)
        lens = MaterialModel(1.6, 55, 0.07)
        vitreous = MaterialModel(1.3, 40, 0.04)

        eye_materials = EyeMaterials(cornea, aqueous, lens, vitreous)

        assert eye_materials.cornea == cornea
        assert eye_materials.aqueous == aqueous
        assert eye_materials.lens == lens
        assert eye_materials.vitreous == vitreous


class TestNavarroMaterials:
    def test_default_initialization(self):
        navarro_materials = NavarroMaterials()
        assert navarro_materials.cornea.refractive_index == 1.3760893927086157
        assert navarro_materials.aqueous.refractive_index == 1.337454140357322
        assert navarro_materials.lens.refractive_index == 1.4201120875715691
        assert navarro_materials.vitreous.refractive_index == 1.3360870349145728
