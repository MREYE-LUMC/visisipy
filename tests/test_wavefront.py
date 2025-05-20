from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.wavefront import ZernikeCoefficients, min_max_noll_index


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
            ("x", 0.0, pytest.raises(TypeError, match="The coefficient must be an integer or a tuple of two integers")),
            (1.5, 0.0, pytest.raises(TypeError, match="The coefficient must be an integer or a tuple of two integers")),
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
            ((0, 0), 0.5, does_not_raise()),
            (2, 0.9, does_not_raise()),
            ((1, 1), 0.9, does_not_raise()),
            (3, 0.0, does_not_raise()),
            ((1, -1), 0.0, does_not_raise()),
            (0, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            (-1, 0.0, pytest.raises(ValueError, match="The coefficient must be larger than 0")),
            ("x", 0.0, pytest.raises(TypeError, match="The coefficient must be an integer or a tuple of two integers")),
            (1.5, 0.0, pytest.raises(TypeError, match="The coefficient must be an integer or a tuple of two integers")),
            ((-1, 0), 0.0, pytest.raises(ValueError, match="n must be greater than or equal to 0")),
            ((0, 1), 0.0, pytest.raises(ValueError, match="m must be less than or equal to n")),
            ((3, 5), 0.0, pytest.raises(ValueError, match="m must be less than or equal to n")),
            ((5, -4), 0.0, pytest.raises(ValueError, match="n and m must have the same parity")),
            ((5, 4), 0.0, pytest.raises(ValueError, match="n and m must have the same parity")),
        ],
    )
    def test_getitem(self, n, expected_value, expectation):
        zernike = ZernikeCoefficients({1: 0.5, 2: 0.9})

        with expectation:
            assert zernike[n] == expected_value

    @pytest.mark.parametrize(
        "nm,noll",
        [
            ((0, 0), 1),
            ((1, 1), 2),
            ((1, -1), 3),
            ((2, 0), 4),
            ((2, -2), 5),
            ((2, 2), 6),
            ((3, -1), 7),
            ((3, 1), 8),
            ((3, -3), 9),
            ((3, 3), 10),
            ((4, 0), 11),
            ((4, 2), 12),
            ((4, -2), 13),
            ((4, 4), 14),
            ((4, -4), 15),
            ((5, 1), 16),
            ((5, -1), 17),
            ((5, 3), 18),
            ((5, -3), 19),
            ((5, 5), 20),
        ],
    )
    def test_to_noll(self, nm, noll):
        assert ZernikeCoefficients.to_noll(*nm) == noll


class TestMinMaxNollIndex:
    @pytest.mark.parametrize("order", range(100))
    def test_consistent_with_to_noll(self, order: int):
        min_index, max_index = min_max_noll_index(order, order)

        assert min_index == min(ZernikeCoefficients.to_noll(order, s * (order % 2)) for s in (-1, 1))
        assert max_index == max(ZernikeCoefficients.to_noll(order, s * order) for s in (-1, 1))

    @pytest.mark.parametrize("order", range(100))
    def test_min_max_adjacent(self, order: int):
        min_index, max_index = min_max_noll_index(order + 1, order)

        assert min_index == max_index + 1
