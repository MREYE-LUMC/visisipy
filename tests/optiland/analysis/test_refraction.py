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


@pytest.fixture
def mock_update_pupil(optiland_backend, monkeypatch):
    monkeypatch.setattr(optiland_backend, "aperture_history", [], raising=False)

    @classmethod
    def update_pupil(cls, pupil_diameter):
        cls.aperture_history.append(pupil_diameter)

    monkeypatch.setattr(optiland_backend, "update_pupil", update_pupil)


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
        "pupil_diameter,change_pupil_diameter",
        [
            (None, False),
            (1.234, True),
        ],
    )
    def test_change_pupil(
        self,
        pupil_diameter,
        change_pupil_diameter,
        optiland_backend,
        mock_update_pupil,
        optiland_analysis,
    ):
        optiland_backend.build_model(EyeModel())
        optiland_backend.update_settings(aperture_type="float_by_stop_size", aperture_value=1.0)

        assert optiland_backend.aperture_history == []

        optiland_analysis.refraction(pupil_diameter=pupil_diameter)

        if change_pupil_diameter:
            assert optiland_backend.aperture_history == [pupil_diameter, 1.0]
        else:
            assert optiland_backend.aperture_history == []

    @pytest.mark.parametrize(
        "aperture_type,pupil_diameter",
        [
            ("entrance_pupil_diameter", 1.0),
            ("image_f_number", 2.0),
            ("object_numeric_aperture", 1e-10),
        ],
    )
    def test_change_pupil_warns_aperture(self, pupil_diameter, aperture_type, optiland_backend, optiland_analysis):
        optiland_backend.build_model(EyeModel())
        optiland_backend.update_settings(aperture_type=aperture_type)

        with pytest.warns(
            UserWarning, match="When updating the pupil size for aperture types other than 'float_by_stop_size'"
        ):
            optiland_analysis.refraction(pupil_diameter=pupil_diameter)
