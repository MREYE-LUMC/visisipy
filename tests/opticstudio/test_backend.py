from contextlib import nullcontext as does_not_raise

import pytest
import zospy as zp

from visisipy import EyeModel
from visisipy.opticstudio.backend import OpticStudioBackend

pytestmark = [pytest.mark.needs_opticstudio]


class TestOpticStudioBackend:
    def test_initialize_opticstudio(self, opticstudio_backend):
        assert opticstudio_backend.zos is not None
        assert opticstudio_backend.oss is not None

    @pytest.mark.parametrize(
        "ray_aiming,ray_aiming_constant,expectation",
        [
            ("off", "Off", does_not_raise()),
            ("paraxial", "Paraxial", does_not_raise()),
            ("real", "Real", does_not_raise()),
            (
                "invalid",
                None,
                pytest.raises(ValueError, match="ray_aiming must be either 'off', 'paraxial', or 'real'"),
            ),
        ],
    )
    def test_new_model(self, opticstudio_backend, ray_aiming, ray_aiming_constant, expectation):
        with expectation:
            OpticStudioBackend.new_model(save_old_model=False, ray_aiming=ray_aiming)

            assert opticstudio_backend.oss.SystemData.RayAiming.RayAiming == zp.constants.process_constant(
                zp.constants.SystemData.RayAimingMethod, ray_aiming_constant
            )
            assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 3

    def test_build_model(self, opticstudio_backend):
        model = EyeModel()

        opticstudio_backend.build_model(model)

        assert opticstudio_backend.model is not None
        assert opticstudio_backend.model.eye_model == model
        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 7

    def test_clear_model(self, opticstudio_backend):
        model = EyeModel()

        opticstudio_backend.build_model(model)
        assert opticstudio_backend.model is not None
        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 7

        opticstudio_backend.clear_model()

        assert opticstudio_backend.model is None
        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 3

    def test_save_model(self, opticstudio_backend, tmp_path):
        model = EyeModel()

        opticstudio_backend.build_model(model)

        # Save the model to a temporary path
        model_path = tmp_path / "model.zmx"
        opticstudio_backend.save_model(model_path)

        assert model_path.exists()

        # Remove the model file
        model_path.unlink()

        assert not model_path.exists()

        # Save without specifying a path
        opticstudio_backend.save_model()

        assert model_path.exists()

    def test_disconnect(self, opticstudio_backend):
        opticstudio_backend.disconnect()

        assert opticstudio_backend.zos is None
        assert opticstudio_backend.oss is None

    @pytest.mark.parametrize(
        "coordinates,field_type,field_constant,expectation",
        [
            ([(10, 10)], "angle", "Angle", does_not_raise()),
            (
                [(0, 0), (0, 10), (-10, 0), (10, -10)],
                "angle",
                "Angle",
                does_not_raise(),
            ),
            (
                [(3.14, 4.15), (-12, 1)],
                "object_height",
                "ObjectHeight",
                does_not_raise(),
            ),
            (
                [(0, 0)],
                "invalid",
                None,
                pytest.raises(
                    ValueError,
                    match="field_type must be either 'angle' or 'object_height'",
                ),
            ),
        ],
    )
    def test_set_fields(self, opticstudio_backend, coordinates, field_type, field_constant, expectation):
        with expectation:
            opticstudio_backend.set_fields(coordinates, field_type)

            assert opticstudio_backend.oss.SystemData.Fields.NumberOfFields == len(coordinates)
            for i in range(len(coordinates)):
                field = opticstudio_backend.oss.SystemData.Fields.GetField(i + 1)
                assert coordinates[i] == (field.X, field.Y)

            assert opticstudio_backend.oss.SystemData.Fields.GetFieldType() == zp.constants.process_constant(
                zp.constants.SystemData.FieldType, field_constant
            )

    def test_set_wavelengths(self, opticstudio_backend):
        opticstudio_backend.set_wavelengths([0.543, 0.650])

        assert opticstudio_backend.oss.SystemData.Wavelengths.NumberOfWavelengths == 2
        assert opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength == 0.543
        assert opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(2).Wavelength == 0.650

    def test_get_wavelength_number(self, opticstudio_backend):
        opticstudio_backend.set_wavelengths([0.543, 0.650])

        assert opticstudio_backend.get_wavelength_number(0.543) == 1
        assert opticstudio_backend.get_wavelength_number(0.650) == 2
        assert opticstudio_backend.get_wavelength_number(1.234) is None
