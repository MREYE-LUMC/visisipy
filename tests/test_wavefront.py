from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.wavefront import ZernikeCoefficients


class TestZernikeCoefficients:
    @pytest.mark.parametrize(
        "terms,result,expectation",
        [
            (None, {}, does_not_raise()),
            ({}, {}, does_not_raise()),
            ({1: 0.5, 2: 0.9}, {1: 0.5, 2: 0.9}, does_not_raise()),
            ({1: 0.5, "x": 0.9}, {}, pytest.raises(TypeError, match="All keys must be integers")),
            ({1: 0.5, 2.8: 0.9}, {}, pytest.raises(TypeError, match="All keys must be integers")),
            ({1: 0.5, 0: 0.9}, {}, pytest.raises(ValueError, match="The Zernike coefficients must be larger than 0")),
            ({-1: 0.5, 2: 0.9}, {}, pytest.raises(ValueError, match="The Zernike coefficients must be larger than 0")),
        ],
    )
    def test_init(self, terms, result, expectation):
        with expectation:
            zernike = ZernikeCoefficients(terms)
            assert zernike == result

    @pytest.mark.parametrize(
        "n,value,expectation",
        [
            (1, 1.2, does_not_raise()),
            (1, 0.0, does_not_raise()),
            (0, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            (-1, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            ("x", 0.0, pytest.raises(TypeError, match="The key must be an integer")),
            (1.5, 0.0, pytest.raises(TypeError, match="The key must be an integer")),
        ],
    )
    def test_setitem(self, n: int, value: float, expectation):
        zernike = ZernikeCoefficients()

        with expectation:
            zernike[n] = value
            assert zernike[n] == value

    @pytest.mark.parametrize(
        "n,expected_value,expectation",
        [
            (1, 0.5, does_not_raise()),
            (2, 0.9, does_not_raise()),
            (3, 0.0, does_not_raise()),
            (0, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            (-1, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            ("x", 0.0, pytest.raises(TypeError, match="The key must be an integer")),
            (1.5, 0.0, pytest.raises(TypeError, match="The key must be an integer")),
        ],
    )
    def test_getitem(self, n, expected_value, expectation):
        zernike = ZernikeCoefficients({1: 0.5, 2: 0.9})

        with expectation:
            assert zernike[n] == expected_value
