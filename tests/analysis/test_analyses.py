from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import visisipy
import visisipy.backend


class MockAnalysis:
    def cardinal_points(
        self,
        surface_1,
        surface_2,
    ):
        return None, None

    def fft_psf(self, field_coordinate, wavelength, field_type, sampling, psf_type):
        return None, None

    def raytrace(
        self,
        coordinates,
        wavelengths,
        field_type,
        pupil,
    ):
        return None, None

    def zernike_standard_coefficients(self, field_coordinate, wavelength, field_type, sampling, maximum_term):
        return None, None

    def refraction(
        self,
        field_coordinate,
        wavelength,
        sampling,
        pupil_diameter,
        field_type,
        *,
        use_higher_order_aberrations,
    ):
        return None, None


class MockBackend:
    analysis = MockAnalysis()
    model = object()


def test_cardinal_points_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.cardinal_points() is None
    assert visisipy.analysis.cardinal_points(return_raw_result=True) == (None, None)


@pytest.mark.parametrize(
    "psf_type,expectation",
    [
        ("linear", does_not_raise()),
        ("logarithmic", does_not_raise()),
        ("invalid", pytest.raises(ValueError, match="psf_type must be either 'linear' or 'logarithmic'")),
    ],
)
def test_fft_psf_analysis(monkeypatch, psf_type, expectation):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.fft_psf() is None
    assert visisipy.analysis.fft_psf(return_raw_result=True) == (None, None)


def test_raytracing_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.raytrace(coordinates=[(0, 0)]) is None
    assert visisipy.analysis.raytrace(coordinates=[(0, 0)], return_raw_result=True) == (
        None,
        None,
    )


def test_zernike_standard_coefficients_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.zernike_standard_coefficients() is None
    assert visisipy.analysis.zernike_standard_coefficients(return_raw_result=True) == (
        None,
        None,
    )


def test_refraction_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.refraction() is None
    assert visisipy.analysis.refraction(return_raw_result=True) == (None, None)
