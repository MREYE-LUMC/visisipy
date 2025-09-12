from __future__ import annotations

import math
import re
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zospy as zp

from visisipy import EyeModel

if TYPE_CHECKING:
    from zospy.zpcore import OpticStudioSystem

    from visisipy.opticstudio.backend import OpticStudioSettings

pytestmark = [pytest.mark.needs_opticstudio]


class TestOpticStudioBackend:
    def test_initialize_opticstudio(self, opticstudio_backend):
        assert opticstudio_backend.zos is not None
        assert opticstudio_backend.oss is not None

    def test_new_model(self, opticstudio_backend):
        # Change a setting and add a new surface
        opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength = 0.640
        opticstudio_backend.oss.LDE.InsertNewSurfaceAt(2)

        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 4

        opticstudio_backend.new_model(save_old_model=False)

        assert (
            opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength
            == opticstudio_backend.settings["wavelengths"][0]
        )
        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 3

    def test_build_model(self, opticstudio_backend):
        model = EyeModel()

        opticstudio_backend.build_model(model)

        assert opticstudio_backend.model is not None
        assert opticstudio_backend.model.eye_model == model
        assert opticstudio_backend.oss.LDE.NumberOfSurfaces == 7

    @pytest.mark.parametrize(
        "aperture_type", ["entrance_pupil_diameter", "float_by_stop_size", "image_f_number", "object_numeric_aperture"]
    )
    def test_build_model_updates_aperture_value(self, opticstudio_backend, aperture_type):
        model = EyeModel()

        # Ensure the aperture value is different from the eye model pupil diameter
        aperture_value = model.geometry.pupil.semi_diameter * 2 + 1
        opticstudio_backend.update_settings(aperture_type=aperture_type, aperture_value=aperture_value)
        opticstudio_backend.build_model(model)

        if aperture_type == "float_by_stop_size":
            # For float by stop size, the value is interpreted as the semi-diameter
            assert opticstudio_backend.get_setting("aperture_value") == model.geometry.pupil.semi_diameter * 2
            assert opticstudio_backend.get_oss().LDE.GetSurfaceAt(3).SemiDiameter == model.geometry.pupil.semi_diameter
            assert opticstudio_backend.get_aperture() == ("float_by_stop_size", model.geometry.pupil.semi_diameter * 2)
        else:
            # For other aperture types, the value is interpreted as the aperture value
            assert opticstudio_backend.get_setting("aperture_value") == aperture_value
            assert opticstudio_backend.get_oss().SystemData.Aperture.ApertureValue == aperture_value
            assert opticstudio_backend.get_aperture() == (aperture_type, aperture_value)

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

    @pytest.mark.parametrize(
        "filename,expectation",
        [
            ("navarro_eye.zmx", does_not_raise()),
            ("nonexistent.zmx", pytest.raises(FileNotFoundError, match="The specified file does not exist:")),
            (
                "navarro_eye.json",
                pytest.raises(
                    ValueError, match=re.escape("File has extension .json, but only .zmx and .zos are supported.")
                ),
            ),
        ],
    )
    def test_load_model(self, opticstudio_backend, datadir, filename, expectation):
        file = datadir / "test_load_models" / filename
        oss: OpticStudioSystem = opticstudio_backend.get_oss()

        with expectation:
            opticstudio_backend.load_model(file, apply_settings=False)

            assert opticstudio_backend.model is None
            assert oss.LDE.NumberOfSurfaces == 7
            assert oss.LDE.GetSurfaceAt(0).Comment == "Test load file"
            assert str(oss.SystemData.Aperture.ApertureType) == "EntrancePupilDiameter"
            assert oss.SystemData.Wavelengths.NumberOfWavelengths == 4
            assert oss.SystemData.Fields.NumberOfFields == 4

    def test_load_model_apply_settings(self, opticstudio_backend, datadir):
        file = datadir / "test_load_models" / "navarro_eye.zmx"
        oss: OpticStudioSystem = opticstudio_backend.get_oss()

        opticstudio_backend.load_model(file, apply_settings=True)

        assert opticstudio_backend.model is None
        assert oss.LDE.NumberOfSurfaces == 7
        assert str(oss.SystemData.Aperture.ApertureType) == "FloatByStopSize"
        assert oss.SystemData.Wavelengths.NumberOfWavelengths == 1
        assert oss.SystemData.Fields.NumberOfFields == 1

    def test_disconnect(self, opticstudio_backend):
        opticstudio_backend.disconnect()

        assert opticstudio_backend.zos is None
        assert opticstudio_backend.oss is None

    @pytest.mark.parametrize(
        "aperture_type,aperture_value",
        [
            ("entrance_pupil_diameter", 3.0),
            ("float_by_stop_size", 1.4),
            ("image_f_number", 1.0),
            ("object_numeric_aperture", 0.1),
        ],
    )
    def test_get_aperture(self, opticstudio_backend, aperture_type, aperture_value):
        opticstudio_backend.update_settings(aperture_type=aperture_type, aperture_value=aperture_value)

        assert opticstudio_backend.get_aperture() == (aperture_type, aperture_value)

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
                [(math.pi, 4.15), (-12, 1)],
                "object_height",
                "ObjectHeight",
                does_not_raise(),
            ),
            (
                [(np.int64(10), np.int64(11))],  # Test float conversion
                "angle",
                "Angle",
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

    def test_get_fields(self, opticstudio_backend):
        coordinates = [(0, 0), (0, 10), (-10, 0), (10, -10)]

        for i, f in enumerate(coordinates, start=1):
            if i == 1:
                field = opticstudio_backend.get_oss().SystemData.Fields.GetField(i)
                field.X, field.Y = f
            else:
                opticstudio_backend.get_oss().SystemData.Fields.AddField(*f, 1)

        assert opticstudio_backend.get_fields() == coordinates

    def test_set_wavelengths(self, opticstudio_backend):
        opticstudio_backend.set_wavelengths([0.543, 0.650])

        assert opticstudio_backend.oss.SystemData.Wavelengths.NumberOfWavelengths == 2
        assert opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(1).Wavelength == 0.543
        assert opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(2).Wavelength == 0.650

    def test_get_wavelengths(self, opticstudio_backend):
        wavelengths = [0.543, 0.650]

        for i, w in enumerate(wavelengths, start=1):
            if i == 1:
                opticstudio_backend.get_oss().SystemData.Wavelengths.GetWavelength(i).Wavelength = w
            else:
                opticstudio_backend.get_oss().SystemData.Wavelengths.AddWavelength(w, i)

        assert opticstudio_backend.get_wavelengths() == wavelengths

    def test_get_wavelength_number(self, opticstudio_backend):
        opticstudio_backend.set_wavelengths([0.543, 0.650])

        assert opticstudio_backend.get_wavelength_number(0.543) == 1
        assert opticstudio_backend.get_wavelength_number(0.650) == 2
        assert opticstudio_backend.get_wavelength_number(1.234) is None

    @pytest.mark.parametrize(
        "aperture_type,new_aperture_value",
        [
            ("entrance_pupil_diameter", 2.5),
            ("float_by_stop_size", 10.0),
            ("image_f_number", 2.0),
            ("object_numeric_aperture", 0.2),
        ],
    )
    def test_update_pupil(self, opticstudio_backend, aperture_type, new_aperture_value):
        opticstudio_backend.update_settings(
            aperture_type=aperture_type,
            aperture_value=1,
        )

        opticstudio_backend.update_pupil(new_aperture_value)

        assert opticstudio_backend.get_aperture() == (aperture_type, new_aperture_value)

        if aperture_type == "float_by_stop_size":
            # For float by stop size, the value is interpreted as the semi-diameter
            assert (
                opticstudio_backend.get_oss()
                .LDE.GetSurfaceAt(opticstudio_backend.get_oss().LDE.StopSurface)
                .SemiDiameter
                == new_aperture_value / 2
            )
        else:
            assert opticstudio_backend.get_oss().SystemData.Aperture.ApertureValue == new_aperture_value


class TestOpticStudioBackendSettings:
    def test_update_settings(self, opticstudio_backend):
        settings: OpticStudioSettings = {
            "field_type": "object_height",
            "fields": [(0, 0), (0, 10), (-10, 0), (10, -10)],
            "ray_aiming": "paraxial",
            "wavelengths": [0.543, 0.650],
            "aperture_type": "entrance_pupil_diameter",
            "aperture_value": 3.0,
        }

        opticstudio_backend.update_settings(**settings)

        assert all(opticstudio_backend.settings[k] == v for k, v in settings.items())

    @pytest.mark.parametrize(
        "field_type,fields,expected_field_type",
        [
            ("angle", [(10, 10)], "Angle"),
            ("object_height", [(0, 0), (0, 10), (-10, 0), (10, -10)], "ObjectHeight"),
        ],
    )
    def test_field(self, field_type, fields, expected_field_type, opticstudio_backend):
        opticstudio_backend.update_settings(field_type=field_type, fields=fields)

        assert opticstudio_backend.oss.SystemData.Fields.NumberOfFields == len(fields)
        for i in range(len(fields)):
            field = opticstudio_backend.oss.SystemData.Fields.GetField(i + 1)
            assert fields[i] == (field.X, field.Y)

        assert opticstudio_backend.oss.SystemData.Fields.GetFieldType() == zp.constants.process_constant(
            zp.constants.SystemData.FieldType, expected_field_type
        )

    @pytest.mark.parametrize(
        "wavelengths",
        [[0.543, 0.650], [0.543, 0.650, 0.450]],
    )
    def test_wavelength(self, wavelengths, opticstudio_backend):
        opticstudio_backend.update_settings(wavelengths=wavelengths)

        assert opticstudio_backend.oss.SystemData.Wavelengths.NumberOfWavelengths == len(wavelengths)
        for i, wavelength in enumerate(wavelengths):
            assert opticstudio_backend.oss.SystemData.Wavelengths.GetWavelength(i + 1).Wavelength == wavelength

    @pytest.mark.parametrize(
        "aperture_type, aperture_value, expected_aperture_type",
        [
            ("entrance_pupil_diameter", 3.0, "EntrancePupilDiameter"),
            ("float_by_stop_size", None, "FloatByStopSize"),
            ("image_f_number", 1.0, "ImageSpaceFNum"),
            ("object_numeric_aperture", 0.1, "ObjectSpaceNA"),
        ],
    )
    def test_aperture(self, aperture_type, aperture_value, expected_aperture_type, opticstudio_backend):
        opticstudio_backend.update_settings(aperture_type=aperture_type, aperture_value=aperture_value)

        assert opticstudio_backend.oss.SystemData.Aperture.ApertureType == zp.constants.process_constant(
            zp.constants.SystemData.ZemaxApertureType, expected_aperture_type
        )

        if aperture_type != "float_by_stop_size":
            assert opticstudio_backend.oss.SystemData.Aperture.ApertureValue == aperture_value

    @pytest.mark.parametrize(
        "ray_aiming,expected",
        [
            ("off", "Off"),
            ("paraxial", "Paraxial"),
            ("real", "Real"),
        ],
    )
    def test_ray_aiming(self, ray_aiming, expected, opticstudio_backend):
        opticstudio_backend.update_settings(ray_aiming=ray_aiming)

        assert opticstudio_backend.oss.SystemData.RayAiming.RayAiming == zp.constants.process_constant(
            zp.constants.SystemData.RayAimingMethod, expected
        )
