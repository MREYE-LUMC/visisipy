from __future__ import annotations

import math
from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from optiland.fields import FieldGroup

    from visisipy import EyeModel
    from visisipy.backend import BackendSettings
    from visisipy.optiland.backend import OptilandBackend


class TestOptilandBackend:
    def test_initialize_optiland(self, optiland_backend: OptilandBackend):
        assert optiland_backend.optic is not None
        assert optiland_backend.model is None

    def test_new_model(self, optiland_backend: OptilandBackend):
        # Change a setting and add a new surface
        optiland_backend.get_optic().add_field(10, 10)
        optiland_backend.get_optic().add_surface(index=0)

        assert optiland_backend.get_optic().surface_group.num_surfaces == 1

        optiland_backend.new_model(save_old_model=False)

        assert optiland_backend.get_optic().surface_group.num_surfaces == 0
        assert optiland_backend.get_optic().fields.num_fields == 1

    def test_build_model(self, optiland_backend: OptilandBackend, eye_model: EyeModel):
        optiland_backend.build_model(eye_model)

        assert optiland_backend.model is not None
        assert optiland_backend.model.eye_model == eye_model
        assert optiland_backend.get_optic().surface_group.num_surfaces == 7

    def test_clear_model(self, optiland_backend: OptilandBackend, eye_model: EyeModel):
        optiland_backend.build_model(eye_model)
        assert optiland_backend.model is not None
        assert optiland_backend.get_optic().surface_group.num_surfaces == 7

        optiland_backend.clear_model()

        assert optiland_backend.model is None
        assert optiland_backend.get_optic().surface_group.num_surfaces == 0

    def test_save_model(self, optiland_backend: OptilandBackend, eye_model, tmp_path):
        optiland_backend.build_model(eye_model)

        path = tmp_path / "test_model.json"
        optiland_backend.save_model(path=path)

        assert path.exists()

    def test_save_model_invalid_filename_raises_valueerror(self, optiland_backend: OptilandBackend):
        with pytest.raises(ValueError, match=r"filename must end in \.json"):
            optiland_backend.save_model(path="invalid_path/test_model.zmx")

    @pytest.mark.parametrize(
        "coordinates,field_type,expectation",
        [
            ([(10, 10)], "angle", does_not_raise()),
            (
                [(0, 0), (0, 10), (-10, 0), (10, -10)],
                "angle",
                does_not_raise(),
            ),
            (
                [(math.pi, 4.15), (-12, 1)],
                "object_height",
                does_not_raise(),
            ),
            (
                [(0, 0)],
                "invalid",
                pytest.raises(
                    ValueError,
                    match="field_type must be either 'angle' or 'object_height'",
                ),
            ),
        ],
    )
    def test_set_fields(self, coordinates, field_type, expectation, optiland_backend: OptilandBackend):
        with expectation:
            optiland_backend.set_fields(coordinates, field_type)

            assert optiland_backend.get_optic().fields.num_fields == len(coordinates)
            for coordinate, field in zip(coordinates, optiland_backend.get_optic().fields.fields):
                assert coordinate == (field.x, field.y)

            assert all(f.field_type == field_type for f in optiland_backend.get_optic().fields.fields)

    def test_get_fields(self, optiland_backend: OptilandBackend):
        coordinates = [(10, 10), (20, 20)]
        field_type = "angle"

        optiland_backend.get_optic().fields.fields.clear()
        optiland_backend.get_optic().set_field_type(field_type)

        for coordinate in coordinates:
            optiland_backend.get_optic().add_field(coordinate[0], coordinate[1])

        assert optiland_backend.get_fields() == coordinates

    def test_set_wavelengths(self, optiland_backend: OptilandBackend):
        wavelengths = [0.543, 0.650, 0.450]

        optiland_backend.set_wavelengths(wavelengths)

        assert optiland_backend.get_optic().wavelengths.num_wavelengths == len(wavelengths)
        assert all(w.value == e for w, e in zip(optiland_backend.get_optic().wavelengths.wavelengths, wavelengths))

    def test_get_wavelengths(self, optiland_backend: OptilandBackend):
        wavelengths = [0.543, 0.650, 0.450]

        optiland_backend.get_optic().wavelengths.wavelengths.clear()

        for wavelength in wavelengths:
            optiland_backend.get_optic().add_wavelength(wavelength)

        assert optiland_backend.get_wavelengths() == wavelengths


class TestOptilandBackendSettings:
    def test_update_settings(self, optiland_backend):
        settings: BackendSettings = {
            "field_type": "object_height",
            "fields": [(0, 0), (0, 10), (-10, 0), (10, -10)],
            "wavelengths": [0.55, 0.65, 0.75],
            "aperture_type": "entrance_pupil_diameter",
            "aperture_value": 3.0,
        }

        optiland_backend.update_settings(**settings)

        assert all(optiland_backend.settings[k] == v for k, v in settings.items())

    @staticmethod
    def _assert_fields_equal(optic_fields: FieldGroup, expected_fields: list[tuple[float, float]]):
        assert len(optic_fields.fields) == len(expected_fields)

        for optiland_field, expected_field in zip(optic_fields.fields, expected_fields):
            assert optiland_field.x == expected_field[0]
            assert optiland_field.y == expected_field[1]

    @pytest.mark.parametrize(
        "field_type,fields",
        [
            ("angle", [(10, 10)]),
            ("object_height", [(0, 0), (0, 10), (-10, 0), (10, -10)]),
        ],
    )
    def test_field(self, field_type, fields, optiland_backend: OptilandBackend):
        field_type = "angle"
        fields = [(10, 10), (20, 20)]

        optiland_backend.update_settings(field_type=field_type, fields=fields)

        self._assert_fields_equal(optiland_backend.get_optic().fields, fields)
        assert all(f.field_type == field_type for f in optiland_backend.get_optic().fields.fields)

    @pytest.mark.parametrize(
        "wavelengths",
        [[0.543, 0.650], [0.543, 0.650, 0.450]],
    )
    def test_wavelength(self, wavelengths, optiland_backend: OptilandBackend):
        optiland_backend.update_settings(wavelengths=wavelengths)

        assert all(w.value == e for w, e in zip(optiland_backend.get_optic().wavelengths.wavelengths, wavelengths))

    @pytest.mark.parametrize(
        "aperture_type,aperture_value,expected_aperture_type,expectation",
        [
            ("entrance_pupil_diameter", 3.0, "EPD", does_not_raise()),
            ("image_f_number", 1.0, "imageFNO", does_not_raise()),
            ("object_numeric_aperture", 0.1, "objectNA", does_not_raise()),
            (
                "float_by_stop_size",
                None,
                "FloatByStopSize",
                pytest.raises(
                    NotImplementedError, match="Aperture type 'float_by_stop_size' is not implemented in Optiland"
                ),
            ),
            (
                "invalid_aperture_type",
                3.0,
                "InvalidApertureType",
                pytest.raises(ValueError, match="Invalid aperture type"),
            ),
        ],
    )
    def test_aperture(
        self, aperture_type, aperture_value, expected_aperture_type, expectation, optiland_backend: OptilandBackend
    ):
        with expectation:
            optiland_backend.update_settings(aperture_type=aperture_type, aperture_value=aperture_value)

            assert optiland_backend.get_optic().aperture.ap_type == expected_aperture_type
            assert optiland_backend.get_optic().aperture.value == aperture_value
