"""Utilities for handling wavefront data."""

from __future__ import annotations

from collections import defaultdict

__all__ = ("ZernikeCoefficients",)


def _is_int_tuple(x: tuple[int, int]) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and all(isinstance(i, int) for i in x)  # noqa: PLR2004


def _validate_coefficient(key: int | tuple[int, int]) -> int:
    if isinstance(key, int):
        if key < 1:
            raise ValueError("The coefficient must be larger than 0.")

        return key

    if _is_int_tuple(key):
        _validate_nm(*key)
        return ZernikeCoefficients.to_noll(*key)

    raise TypeError("The coefficient must be an integer or a tuple of two integers.")


def _validate_nm(n: int, m: int) -> None:
    if n < 0:
        raise ValueError("n must be greater than or equal to 0.")
    if abs(m) > n:
        raise ValueError("m must be less than or equal to n.")
    if (n - abs(m)) % 2 != 0:
        raise ValueError("n and m must have the same parity.")


class ZernikeCoefficients(defaultdict):
    """Zernike coefficients.

    Convenience class for handling Zernike coefficients as a dictionary. If a term is not present, 0 is returned.
    Upon initialization and setting items, the keys are validated to be non-negative integers.

    Raises
    ------
    TypeError
        If the key is not an integer.
    ValueError
        If the key is smaller than 1.
    """

    def __init__(self, terms: dict[int, float] | None = None):
        if terms is not None:
            if not all(isinstance(key, int) for key in terms):
                raise TypeError("All keys must be integers.")
            if any(key < 1 for key in terms):
                raise ValueError("The Zernike coefficients must be larger than 0.")

        super().__init__(lambda: 0, terms or {})

    @staticmethod
    def to_noll(n: int, m: int):
        """Convert Zernike indices to Noll indices.

        Parameters
        ----------
        n : int
            The radial order of the Zernike polynomial.
        m : int
            The azimuthal order of the Zernike polynomial.

        Returns
        -------
        int
            The Noll index corresponding to the given Zernike indices.
        """
        # Validate n and m
        _validate_nm(n, m)

        i = n % 4 in {0, 1}
        k = 0 if (m > 0 and i) or (m < 0 and not i) else 1

        return ((n + 1) * n) // 2 + abs(m) + k

    def __setitem__(self, key: int | tuple[int, int], value: float) -> None:
        key = _validate_coefficient(key)

        super().__setitem__(key, value)

    def __getitem__(self, key: int | tuple[int, int]) -> float:
        key = _validate_coefficient(key)

        return super().__getitem__(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"


def min_max_noll_index(min_order: int, max_order: int) -> tuple[int, int]:
    """Get the minimum and maximum Noll indices for a given range of Zernike orders.

    Parameters
    ----------
    min_order : int
        The minimum Zernike order.
    max_order : int
        The maximum Zernike order.

    Returns
    -------
    tuple[int, int]
        The minimum and maximum Noll indices corresponding to the given Zernike orders.
    """
    min_index = ((min_order + 1) * min_order) // 2 + 1
    max_index = ((max_order + 1) * max_order) // 2 + max_order + 1

    return min_index, max_index
