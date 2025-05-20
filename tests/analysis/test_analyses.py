from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import visisipy
import visisipy.backend
from visisipy.wavefront import ZernikeCoefficients


class MockAnalysis:
    def cardinal_points(
        self,
        surface_1,
        surface_2,
    ):
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
        return ZernikeCoefficients({1: 0, 2: 0, 3: 0}), None

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


def test_raytracing_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.raytrace(coordinates=[(0, 0)]) is None
    assert visisipy.analysis.raytrace(coordinates=[(0, 0)], return_raw_result=True) == (
        None,
        None,
    )


def test_zernike_standard_coefficients_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.zernike_standard_coefficients() == ZernikeCoefficients({1: 0, 2: 0, 3: 0})
    assert visisipy.analysis.zernike_standard_coefficients(return_raw_result=True) == (
        ZernikeCoefficients({1: 0, 2: 0, 3: 0}),
        None,
    )


def test_refraction_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.refraction() is None
    assert visisipy.analysis.refraction(return_raw_result=True) == (None, None)


def test_rms_hoa_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.rms_hoa() == 0
    assert visisipy.analysis.rms_hoa(return_raw_result=True) == (0, None)


class TestRMSHOAAnalysis:
    @pytest.fixture(params=["opticstudio", "optiland"])
    def backend(self, request):
        fixture_name = request.param + "_backend"

        return request.getfixturevalue(fixture_name)

    @pytest.mark.parametrize(
        "min_order,max_order,maximum_term,expectation",
        [
            (3, 8, None, does_not_raise()),
            (
                -1,
                8,
                None,
                pytest.raises(ValueError, match="min_order and max_order must be greater than or equal to 0"),
            ),
            (
                0,
                -1,
                None,
                pytest.raises(ValueError, match="min_order and max_order must be greater than or equal to 0"),
            ),
            (0, 0, None, pytest.raises(ValueError, match="max_order must be greater than min_order")),
            (
                3,
                9,
                45,
                pytest.raises(
                    ValueError, match="maximum_term must be greater than or equal to the largest term of max_order"
                ),
            ),
        ],
    )
    def test_rms_hoa_analysis(self, min_order, max_order, maximum_term, expectation, backend, eye_model):
        backend.build_model(eye_model)

        with expectation:
            visisipy.analysis.rms_hoa(
                min_order=min_order,
                max_order=max_order,
                maximum_term=maximum_term,
                backend=backend,
            )
