from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING

import pytest
from optiland.fields import FieldGroup

from visisipy.optiland.backend import OptilandBackend

if TYPE_CHECKING:
    from visisipy.backend import BackendSettings



class TestOptilandBackend:
    def test_initialize_optiland(self, optiland_backend):
        assert optiland_backend.optic is not None
        assert optiland_backend.model is None


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
    def _assert_fields_equal(
        optic_fields: FieldGroup, expected_fields: list[tuple[float, float]]
    ):
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

        self._assert_fields_equal(
            optiland_backend.get_optic().fields, fields
        )
        assert all(
            f.field_type == field_type
            for f in optiland_backend.get_optic().fields.fields
        )

    @pytest.mark.parametrize(
        "wavelengths",
        [[0.543, 0.650], [0.543, 0.650, 0.450]],
    )
    def test_wavelength(self, wavelengths, optiland_backend: OptilandBackend):
        optiland_backend.update_settings(wavelengths=wavelengths)

        assert all(
            w.value == e for w, e in zip(optiland_backend.get_optic().wavelengths.wavelengths, wavelengths)
        )

    @pytest.mark.parametrize(
        "aperture_type,aperture_value,expected_aperture_type,expectation",
        [
            ("entrance_pupil_diameter", 3.0, "EPD", does_not_raise()),
            ("image_f_number", 1.0, "imageFNO", does_not_raise()),
            ("object_numeric_aperture", 0.1, "objectNA", does_not_raise()),
            ("float_by_stop_size", None, "FloatByStopSize", pytest.raises(NotImplementedError, match="Aperture type \'float_by_stop_size\' is not implemented in Optiland")),
            (
                "invalid_aperture_type",
                3.0,
                "InvalidApertureType",
                pytest.raises(ValueError, match="Invalid aperture type"),
            ),
        ],
    )
    def test_aperture(self, aperture_type, aperture_value, expected_aperture_type, expectation, optiland_backend: OptilandBackend):
        with expectation:
            optiland_backend.update_settings(aperture_type=aperture_type, aperture_value=aperture_value)

            assert optiland_backend.get_optic().aperture.ap_type == expected_aperture_type
            assert optiland_backend.get_optic().aperture.value == aperture_value
