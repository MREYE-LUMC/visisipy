from __future__ import annotations

import math
from types import MethodType, SimpleNamespace

import pytest
from optiland.surfaces import Surface

from tests.helpers import build_args
from visisipy.models import EyeModel
from visisipy.types import SampleSize


class MockSurface:
    def __init__(self):
        self.semi_aperture = 1.0
        self.changed_semi_aperture = False

    @property
    def semi_aperture(self):
        return self._semi_aperture

    @semi_aperture.setter
    def semi_aperture(self, value):
        self.changed_semi_aperture = True
        self._semi_aperture = value


class MockOptilandModel:
    def __init__(self):
        self._pupil = SimpleNamespace(surface=MockSurface())

    @property
    def pupil(self):
        return self._pupil


def patch_surface(surface: Surface, monkeypatch):
    def set_semi_aperture(self, a):
        self.changed_semi_aperture = True
        self.semi_aperture = a

    monkeypatch.setattr(surface, "set_semi_aperture", MethodType(set_semi_aperture, Surface))
    monkeypatch.setattr(surface, "changed_semi_aperture", False, raising=False)


class TestRefractionAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,sampling,pupil_diameter,field_type,use_higher_order_aberrations",
        [
            (None, None, None, None, None, None),
            ((0, 0), 0.543, None, None, None, None),
            ((1, 1), 0.632, 64, None, None, None),
            ((math.pi, math.tau), 0.543, "64x64", 0.5, None, None),
            ((0, 0), 0.543, "64x64", 0.5, "angle", None),
            pytest.param(
                (1, 1),
                0.632,
                SampleSize(32),
                0.5,
                "object_height",
                None,
            ),
            ((0, 0), 0.543, "32x32", 0.5, "angle", True),
            pytest.param(
                (1, 1),
                0.632,
                "64x64",
                0.5,
                "object_height",
                False,
            ),
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
        optiland_backend,
        optiland_analysis,
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            sampling=sampling,
            pupil_diameter=pupil_diameter,
            field_type=field_type,
            use_higher_order_aberrations=use_higher_order_aberrations,
            non_null_defaults={
                "sampling",
                "field_type",
                "use_higher_order_aberrations",
            },
        )

        assert optiland_analysis.refraction(**args)

    @pytest.mark.parametrize(
        "pupil_diameter,changed_pupil_diameter",
        [
            (None, False),
            (0.5, True),
        ],
    )
    def test_refraction_change_pupil(
        self,
        pupil_diameter,
        changed_pupil_diameter,
        optiland_backend,
        optiland_analysis,
    ):
        optiland_backend.build_model(EyeModel())

        # Manually set the aperture to a sentinel value
        new_semi_aperture = object()
        optiland_analysis.backend.model.pupil.surface.set_semi_aperture(new_semi_aperture)

        assert optiland_analysis.backend.model.pupil.surface.semi_aperture is new_semi_aperture

        optiland_analysis.refraction(pupil_diameter=pupil_diameter)

        if changed_pupil_diameter:
            assert optiland_analysis.backend.model.pupil.surface.semi_aperture == pupil_diameter / 2
        else:
            assert optiland_analysis.backend.model.pupil.surface.semi_aperture is new_semi_aperture
