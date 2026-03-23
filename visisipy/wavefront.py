"""Utilities for handling wavefront data."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    msg = f"The coefficient must be an integer or a tuple of two integers, got {key} of type {type(key).__name__}."
    raise TypeError(msg)


def _validate_nm(n: int, m: int) -> None:
    if n < 0:
        raise ValueError("n must be greater than or equal to 0.")
    if abs(m) > n:
        raise ValueError("m must be less than or equal to n.")
    if (n - abs(m)) % 2 != 0:
        raise ValueError("n and m must have the same parity.")


class ZernikeCoefficients(defaultdict[int, float]):
    """Zernike coefficients.

    Convenience class for handling Zernike coefficients as a dictionary. If a term is not present, 0 is returned.
    Upon initialization and setting items, the keys are validated to be either positive integers or tuples of two integers.
    """

    def __init__(self, terms: Mapping[int, float] | Mapping[tuple[int, int], float] | None = None):
        """Initialize the Zernike coefficients.

        Zernike coefficients can be initialized with either Noll indices (integers) or Zernike indices (tuples of two integers).

        Parameters
        ----------
        terms : Mapping[int, float] | Mapping[tuple[int, int], float] | None, optional
            A mapping of Zernike coefficients, where the keys are either Noll indices (integers)
            or Zernike indices (tuples of two integers), and the values are the corresponding coefficients.
            If None, an empty dictionary is used.

        Raises
        ------
        ValueError
            If any key in the terms mapping is not a valid Zernike index.

        Examples
        --------
        >>> ZernikeCoefficients({(0, 0): 1.0, (1, 1): 0.5, (1, -1): 0.5})
        ZernikeCoefficients({1: 1.0, 2: 0.5, 3: 0.5})
        >>> ZernikeCoefficients({1: 1.0, 2: 0.5, 3: 0.5})
        ZernikeCoefficients({1: 1.0, 2: 0.5, 3: 0.5})
        """
        normalized_terms = {}

        if terms is not None:
            for key, value in terms.items():
                noll_index = _validate_coefficient(key)

                if noll_index in normalized_terms:
                    msg = f"Duplicate coefficient for Noll index {noll_index}."
                    raise ValueError(msg)

                normalized_terms[noll_index] = value

        super().__init__(self._default_factory, normalized_terms)

    @staticmethod
    def _default_factory() -> float:
        return 0.0

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
