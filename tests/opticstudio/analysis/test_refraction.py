from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel
from visisipy.types import SampleSize

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

pytestmark = [pytest.mark.needs_opticstudio]


class TestRefractionAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,sampling,pupil_diameter,field_type,use_higher_order_aberrations",
        [
            (None, None, None, None, None, None),
            ((0, 0), 0.543, None, None, None, None),
            ((1, 1), 0.632, 64, None, None, None),
            ((math.pi, math.tau), 0.543, "64x64", 0.5, None, None),
            ((0, 0), 0.543, "64x64", 0.5, "angle", None),
            ((1, 1), 0.632, SampleSize(32), 0.5, "object_height", None),
            ((0, 0), 0.543, "32x32", 0.5, "angle", True),
            ((1, 1), 0.632, "64x64", 0.5, "object_height", False),
        ],
    )
    def test_refraction(
        self,
        field_coordinate,
        wavelength,
        sampling,
        pupil_diameter,
        field_type,
        use_higher_order_aberrations,
        opticstudio_analysis,
        opticstudio_backend,
        monkeypatch,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            sampling=sampling,
            pupil_diameter=pupil_diameter,
            field_type=field_type,
            use_higher_order_aberrations=use_higher_order_aberrations,
            non_null_defaults={"sampling", "field_type", "use_higher_order_aberrations"},
        )

        assert opticstudio_analysis.refraction(**args)

    @pytest.mark.parametrize(
        "pupil_diameter,change_pupil_diameter",
        [
            (None, False),
            (1.234, True),
        ],
    )
    def test_change_pupil(
        self, opticstudio_backend, opticstudio_analysis, pupil_diameter, change_pupil_diameter, mocker: MockerFixture
    ):
        opticstudio_backend.update_settings(aperture_type="float_by_stop_size", aperture_value=1.0)

        spy = mocker.spy(opticstudio_backend, "update_pupil")

        opticstudio_analysis.refraction(pupil_diameter=pupil_diameter)

        if change_pupil_diameter:
            assert spy.call_count == 2
            assert spy.call_args_list[0] == mocker.call(pupil_diameter)
            assert spy.call_args_list[1] == mocker.call(1.0)
        else:
            spy.assert_not_called()

    @pytest.mark.parametrize(
        "aperture_type,pupil_diameter",
        [
            ("entrance_pupil_diameter", 1.0),
            ("image_f_number", 2.0),
            ("object_numeric_aperture", 0.1),
        ],
    )
    def test_change_pupil_warns_aperture(
        self, pupil_diameter, aperture_type, opticstudio_backend, opticstudio_analysis
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if aperture_type == "object_numeric_aperture" else float("inf")
        )
        opticstudio_backend.update_settings(aperture_type=aperture_type)

        with pytest.warns(
            UserWarning, match="When updating the pupil size for aperture types other than 'float_by_stop_size'"
        ):
            opticstudio_analysis.refraction(pupil_diameter=pupil_diameter)
