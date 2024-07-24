"""Utilities for handling wavefront data."""

from __future__ import annotations

from collections import defaultdict

__all__ = ("ZernikeCoefficients",)


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
    def _validate_coefficient(key: int) -> None:
        if not isinstance(key, int):
            raise TypeError("The key must be an integer.")
        if key < 1:
            raise ValueError("The coefficient must be larger than 0.")

    def __setitem__(self, key: int, value: float) -> None:
        self._validate_coefficient(key)

        super().__setitem__(key, value)

    def __getitem__(self, key: int) -> float:
        self._validate_coefficient(key)

        return super().__getitem__(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self)})"
