from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from visisipy.optiland.analysis.helpers import set_field, set_wavelength

if TYPE_CHECKING:
    from optiland.optic import Optic

    from visisipy.optiland.backend import OptilandBackend
    from visisipy.types import FieldType


def get_optic_wavelengths(optic: Optic) -> list[float]:
    return [w.value for w in optic.wavelengths.wavelengths]


class TestSetWavelength:
    def test_set_wavelength_none(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        default_wavelength = optic.wavelengths.wavelengths[0].value

        assert set_wavelength(optiland_backend, wavelength=None) == default_wavelength

    def test_set_wavelength_existing(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        assert get_optic_wavelengths(optic) == [0.543]
        extra_wavelength_1 = 0.430
        extra_wavelength_2 = 0.320
        optic.wavelengths.add(extra_wavelength_1)
        optic.wavelengths.add(extra_wavelength_2)

        assert set_wavelength(optiland_backend, wavelength=extra_wavelength_1) == extra_wavelength_1
        assert get_optic_wavelengths(optic) == [0.543, extra_wavelength_1, extra_wavelength_2]

    def test_set_wavelength_new(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        assert get_optic_wavelengths(optic) == [0.543]
        new_wavelength = 0.430

        with pytest.warns(UserWarning, match=f"Wavelength {new_wavelength} not found. Adding it to the system."):
            assert set_wavelength(optiland_backend, wavelength=new_wavelength) == new_wavelength

        assert get_optic_wavelengths(optic) == [0.543, new_wavelength]


def get_optic_fields(optic: Optic) -> list[tuple[float, float]]:
    return [(f.x, f.y) for f in optic.fields.fields]


class TestSetField:
    def test_set_field_none(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        default_field = optic.fields.get_field_coords()[0]

        assert set_field(optiland_backend, field_coordinate=None) == default_field

    def test_set_field_existing(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        assert get_optic_fields(optic) == [(0.0, 0.0)]
        extra_field_1 = (10.0, 0.0)
        extra_field_1_normalized = (1.0, 0.0)
        extra_field_2 = (0.0, 10.0)
        optic.fields.add(x=extra_field_1[0], y=extra_field_1[1])
        optic.fields.add(x=extra_field_2[0], y=extra_field_2[1])

        assert set_field(optiland_backend, field_coordinate=extra_field_1) == extra_field_1_normalized
        assert get_optic_fields(optic) == [(0.0, 0.0), extra_field_1, extra_field_2]

    def test_set_field_new(self, optiland_backend: OptilandBackend):
        optic = optiland_backend.optic
        assert get_optic_fields(optic) == [(0.0, 0.0)]
        new_field = (10.0, 0.0)
        new_field_normalized = (1.0, 0.0)

        with pytest.warns(
            UserWarning, match=re.escape(f"Field coordinate {new_field} not found. Adding it to the system.")
        ):
            assert set_field(optiland_backend, field_coordinate=new_field) == new_field_normalized

        assert get_optic_fields(optic) == [(0.0, 0.0), new_field]

    @pytest.mark.parametrize(
        "old_type,new_type,field,normalized_field",
        [
            ("angle", "object_height", (0.0, 0.0), (0.0, 0.0)),
            ("object_height", "angle", (0.0, 0.0), (0.0, 0.0)),
            ("angle", "object_height", None, (0.0, 0.0)),
        ],
    )
    def test_update_field_type(
        self,
        old_type: FieldType,
        new_type: FieldType,
        field: tuple[float, float] | None,
        normalized_field: tuple[float, float],
        optiland_backend: OptilandBackend,
    ):
        optic = optiland_backend.optic
        optic.fields.set_type(old_type)
        assert optiland_backend.get_field_type() == old_type

        with pytest.warns(UserWarning, match=re.escape(f"Changing field type from {old_type} to {new_type}.")):
            assert set_field(optiland_backend, field_coordinate=field, field_type=new_type) == normalized_field

        assert optiland_backend.get_field_type() == new_type
