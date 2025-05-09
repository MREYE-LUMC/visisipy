from __future__ import annotations

import math

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel
from visisipy.types import SampleSize

pytestmark = [pytest.mark.needs_opticstudio]


class MockPupil:
    def __init__(self):
        self.semi_diameter = 1.0
        self.changed_semi_diameter = False

    @property
    def semi_diameter(self):
        return self._semi_diameter

    @semi_diameter.setter
    def semi_diameter(self, value):
        self.changed_semi_diameter = True
        self._semi_diameter = value


class MockOpticstudioModel:
    def __init__(self):
        self._pupil = MockPupil()

    @property
    def pupil(self):
        return self._pupil


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
        # monkeypatch.setattr(opticstudio_analysis.backend, "model", MockOpticstudioModel())

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
        "pupil_diameter,changed_pupil_diameter",
        [
            (None, False),
            (0.5, True),
        ],
    )
    def test_refraction_change_pupil(self, opticstudio_analysis, pupil_diameter, changed_pupil_diameter, monkeypatch):
        monkeypatch.setattr(opticstudio_analysis.backend, "model", MockOpticstudioModel())

        assert not opticstudio_analysis.backend.model.pupil.changed_semi_diameter

        opticstudio_analysis.refraction(pupil_diameter=pupil_diameter)

        assert opticstudio_analysis.backend.model.pupil.changed_semi_diameter == changed_pupil_diameter
        assert opticstudio_analysis.backend.model.pupil.semi_diameter == 1.0
