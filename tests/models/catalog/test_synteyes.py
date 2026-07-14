from __future__ import annotations

import pytest

from tests.test_synteyes import sample_synteye  # noqa: F401
from visisipy.models.catalog.synteyes import SyntEyesEyeModel, SyntEyesGeometry
from visisipy.models.materials import EyeMaterials


class TestSyntEyesGeometry:
    def test_create_synteyes_geometry(self, sample_synteye):  # noqa: F811
        geometry = SyntEyesGeometry(synteye=sample_synteye)

        assert geometry.cornea_front.thickness == sample_synteye.biometry.cornea_thickness
        assert geometry.cornea_front.radius == float("inf")
        assert geometry.cornea_front.zernike_coefficients == sample_synteye.cornea.anterior_zernikes
        assert geometry.cornea_front.norm_radius == sample_synteye.cornea.anterior_norm_diameter / 2

        assert geometry.cornea_back.thickness == sample_synteye.biometry.anterior_chamber_depth
        assert geometry.cornea_back.radius == float("inf")
        assert geometry.cornea_back.zernike_coefficients == sample_synteye.cornea.posterior_zernikes
        assert geometry.cornea_back.norm_radius == sample_synteye.cornea.posterior_norm_diameter / 2

        assert geometry.pupil.semi_diameter == sample_synteye.biometry.pupil_diameter / 2

        assert geometry.lens_front.thickness == sample_synteye.biometry.lens_thickness
        assert geometry.lens_front.radius == sample_synteye.lens.anterior_radius
        assert geometry.lens_front.asphericity == sample_synteye.lens.anterior_conic
        assert geometry.lens_front.zernike_coefficients == sample_synteye.lens.anterior_zernikes
        assert geometry.lens_front.norm_radius == sample_synteye.lens.anterior_norm_diameter / 2

        assert geometry.lens_back.thickness == sample_synteye.biometry.vitreous_depth
        assert geometry.lens_back.radius == sample_synteye.lens.posterior_radius
        assert geometry.lens_back.asphericity == sample_synteye.lens.posterior_conic

        assert geometry.retina.ellipsoid_radii.x == pytest.approx(sample_synteye.retina.radius_x)
        assert geometry.retina.ellipsoid_radii.y == pytest.approx(sample_synteye.retina.radius_y)
        assert geometry.retina.ellipsoid_radii.z == pytest.approx(sample_synteye.retina.radius_z)


class TestSyntEyesEyeModel:
    def test_create_synteyes_eye_model(self, sample_synteye):  # noqa: F811
        model = SyntEyesEyeModel(synteye=sample_synteye)

        assert isinstance(model.geometry, SyntEyesGeometry)
        assert model.geometry.cornea_front.zernike_coefficients == sample_synteye.cornea.anterior_zernikes
        assert model.geometry.cornea_back.zernike_coefficients == sample_synteye.cornea.posterior_zernikes
        assert model.geometry.lens_front.zernike_coefficients == sample_synteye.lens.anterior_zernikes
        assert model.geometry.lens_back.thickness == sample_synteye.biometry.vitreous_depth
        assert model.geometry.retina.ellipsoid_radii.x == pytest.approx(sample_synteye.retina.radius_x)
        assert model.geometry.retina.ellipsoid_radii.y == pytest.approx(sample_synteye.retina.radius_y)
        assert model.geometry.retina.ellipsoid_radii.z == pytest.approx(sample_synteye.retina.radius_z)

    def test_create_random_eye_model(self):
        model = SyntEyesEyeModel()

        assert isinstance(model.geometry, SyntEyesGeometry)
        assert isinstance(model.materials, EyeMaterials)

    def test_create_multiple_random_eye_models(self):
        models = SyntEyesEyeModel.generate(5)

        for model in models:
            assert isinstance(model, SyntEyesEyeModel)
            assert isinstance(model.geometry, SyntEyesGeometry)
            assert isinstance(model.materials, EyeMaterials)

    @pytest.mark.parametrize("num_models", [0, -1])
    def test_generate_invalid_number_of_models(self, num_models):
        with pytest.raises(ValueError, match="n must be a positive integer"):
            SyntEyesEyeModel.generate(num_models)
