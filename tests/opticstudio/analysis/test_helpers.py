from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
import zospy as zp

from visisipy.opticstudio.analysis.helpers import set_field, set_wavelength

if TYPE_CHECKING:
    from zospy.zpcore import OpticStudioSystem

    from visisipy.opticstudio.backend import OpticStudioBackend
    from visisipy.types import FieldType


def get_oss_wavelengths(oss: OpticStudioSystem) -> list[float]:
    return [
        oss.SystemData.Wavelengths.GetWavelength(i).Wavelength
        for i in range(1, oss.SystemData.Wavelengths.NumberOfWavelengths + 1)
    ]


class TestSetWavelength:
    def test_set_wavelength_none(self, opticstudio_backend: type[OpticStudioBackend]):
        default_wavelength_number = 1

        assert set_wavelength(opticstudio_backend, wavelength=None) == default_wavelength_number

    def test_set_wavelength_existing(self, opticstudio_backend: type[OpticStudioBackend]):
        oss = opticstudio_backend.get_oss()
        assert get_oss_wavelengths(oss) == [0.543]
        extra_wavelength_1 = 0.430
        extra_wavelength_2 = 0.320
        oss.SystemData.Wavelengths.AddWavelength(extra_wavelength_1, 1)
        oss.SystemData.Wavelengths.AddWavelength(extra_wavelength_2, 1)

        assert set_wavelength(opticstudio_backend, wavelength=extra_wavelength_1) == 2
        assert get_oss_wavelengths(oss) == [0.543, extra_wavelength_1, extra_wavelength_2]

    def test_set_wavelength_new(self, opticstudio_backend: type[OpticStudioBackend]):
        oss = opticstudio_backend.get_oss()
        assert get_oss_wavelengths(oss) == [0.543]
        new_wavelength = 0.430

        with pytest.warns(UserWarning, match=f"Wavelength {new_wavelength} not found. Adding it to the system."):
            assert set_wavelength(opticstudio_backend, wavelength=new_wavelength) == 2
            assert get_oss_wavelengths(oss) == [0.543, new_wavelength]


def get_oss_fields(oss: OpticStudioSystem) -> list[tuple[float, float]]:
    return [
        (oss.SystemData.Fields.GetField(i).X, oss.SystemData.Fields.GetField(i).Y)
        for i in range(1, oss.SystemData.Fields.NumberOfFields + 1)
    ]


class TestSetField:
    def test_set_field_none(self, opticstudio_backend: type[OpticStudioBackend]):
        default_field_number = 1

        assert set_field(opticstudio_backend, field_coordinate=None) == default_field_number

    def test_set_field_existing(self, opticstudio_backend: type[OpticStudioBackend]):
        oss = opticstudio_backend.get_oss()
        assert get_oss_fields(oss) == [(0.0, 0.0)]
        extra_field_1 = (0.0, 10.0)
        extra_field_2 = (10.0, 0.0)
        oss.SystemData.Fields.AddField(extra_field_1[0], extra_field_1[1], 1)
        oss.SystemData.Fields.AddField(extra_field_2[0], extra_field_2[1], 1)

        assert set_field(opticstudio_backend, field_coordinate=extra_field_1) == 2
        assert get_oss_fields(oss) == [(0.0, 0.0), extra_field_1, extra_field_2]

    def test_set_field_new(self, opticstudio_backend: type[OpticStudioBackend]):
        oss = opticstudio_backend.get_oss()
        assert get_oss_fields(oss) == [(0.0, 0.0)]
        new_field = (10.0, 0.0)

        with pytest.warns(
            UserWarning, match=re.escape(f"Field coordinate {new_field} not found. Adding it to the system.")
        ):
            assert set_field(opticstudio_backend, field_coordinate=new_field) == 2
            assert get_oss_fields(oss) == [(0.0, 0.0), new_field]

    @pytest.mark.parametrize(
        "old_type,new_type,field,field_number",
        [
            ("angle", "object_height", (0, 0), 1),
            ("object_height", "angle", (0, 0), 1),
            ("angle", "object_height", (10, 10), 2),
            ("object_height", "angle", None, 1),
        ],
    )
    def test_update_field_type(
        self,
        old_type: FieldType,
        new_type: FieldType,
        field: tuple[float, float] | None,
        field_number: int,
        opticstudio_backend: type[OpticStudioBackend],
    ):
        oss = opticstudio_backend.get_oss()

        if old_type == "angle":
            oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
        elif old_type == "object_height":
            oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.ObjectHeight)

        assert opticstudio_backend.get_field_type() == old_type

        with pytest.warns(UserWarning, match=f"Changing field type from {old_type} to {new_type}."):
            assert set_field(opticstudio_backend, field_coordinate=field, field_type=new_type) == field_number
            assert opticstudio_backend.get_field_type() == new_type
