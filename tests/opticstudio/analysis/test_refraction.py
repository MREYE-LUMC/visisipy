from __future__ import annotations

import math

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel
from visisipy.types import SampleSize

pytestmark = [pytest.mark.needs_opticstudio]


class MockPupil:
    def __init__(self):
        self.semi_diameter_history = []
        self.semi_diameter = 0.5

    @property
    def semi_diameter(self):
        return self._semi_diameter

    @semi_diameter.setter
    def semi_diameter(self, value):
        self.semi_diameter_history.append(value)
        self._semi_diameter = value


class MockOpticstudioModel:
    def __init__(self):
        self._pupil = MockPupil()

    @property
    def pupil(self):
        return self._pupil


@pytest.fixture
def mock_update_pupil(opticstudio_backend, monkeypatch):
    monkeypatch.setattr(opticstudio_backend, "aperture_history", [], raising=False)

    @classmethod
    def update_pupil(cls, pupil_diameter):
        cls.aperture_history.append(pupil_diameter)

    monkeypatch.setattr(opticstudio_backend, "update_pupil", update_pupil)


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
        self, opticstudio_backend, mock_update_pupil, opticstudio_analysis, pupil_diameter, change_pupil_diameter
    ):
        opticstudio_backend.update_settings(aperture_type="float_by_stop_size", aperture_value=1.0)

        assert opticstudio_backend.aperture_history == []

        opticstudio_analysis.refraction(pupil_diameter=pupil_diameter)

        if change_pupil_diameter:
            assert opticstudio_backend.aperture_history == [pupil_diameter, 1.0]
        else:
            assert opticstudio_backend.aperture_history == []

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
