"""Types for use throughout visisipy.

This module contains type definitions and utility classes for use throughout the visisipy package.
"""

from __future__ import annotations

import re
import sys
from typing import Literal

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict, Unpack
else:
    from typing import NotRequired, TypedDict, Unpack


__all__ = ("ApertureType", "FieldCoordinate", "FieldType", "NotRequired", "SampleSize", "TypedDict", "Unpack")

ApertureType = Literal[
    "float_by_stop_size",
    "entrance_pupil_diameter",
    "image_f_number",
    "object_numeric_aperture",
]
FieldType = Literal["angle", "object_height"]
FieldCoordinate = tuple[float, float]


RE_SAMPLE_SIZE = re.compile(r"^(?P<sampling>\d+)x(?P=sampling)$", re.IGNORECASE)


class SampleSize:
    """Sample size.

    This class represents a sample size used for various analyses.
    How the sample size is used depends on the backend. Sample sizes can be specified as integers or strings in the
    format "NxN", where N is an integer. Only symmetric sample sizes are supported.

    Attributes
    ----------
    sampling : int
        The sample size.

    Example
    -------
    >>> sample_size = SampleSize(512)
    >>> print(int(sample_size))
    512
    >>> sample_size = SampleSize("512x512")
    >>> print(sample_size)
    512x512
    """

    __slots__ = ("__sampling",)

    def __init__(self, sample_size: int | str | SampleSize):
        """Create a new sample size.

        The sample size can be specified as an integer or a string in the format "NxN", where N is an integer.

        Parameters
        ----------
        sample_size : int | str | SampleSize
            The sample size. Can be an integer, a string in the format "NxN", or a SampleSize object.
        """
        if isinstance(sample_size, int):
            self.__sampling = sample_size

        elif isinstance(sample_size, str):
            match = RE_SAMPLE_SIZE.match(sample_size)

            if not match:
                raise ValueError(f"Invalid sample size format: {sample_size}")

            self.__sampling = int(match.group("sampling"))

        elif isinstance(sample_size, SampleSize):
            self.__sampling = sample_size.sampling

    @property
    def sampling(self) -> int:
        """Get the sample size."""
        return self.__sampling

    def __int__(self):
        return self.sampling

    def __str__(self):
        return f"{self.sampling}x{self.sampling}"

    def __repr__(self):
        return f"SampleSize({self.sampling})"

    def __mul__(self, factor: int) -> SampleSize:
        """Multiply the sample size by an integer.

        Parameters
        ----------
        factor : int
            The integer to multiply the sample size by.

        Returns
        -------
        SampleSize
            A new SampleSize object with the multiplied sample size.
        """
        if not isinstance(factor, int):
            raise TypeError("SampleSize can only be multiplied by integer factors.")
        if factor <= 0:
            raise ValueError("SampleSize factor must be a positive integer.")

        return SampleSize(self.sampling * factor)

    def __rmul__(self, factor: int) -> SampleSize:
        """Multiply the sample size by an integer.

        Parameters
        ----------
        factor : int
            The integer to multiply the sample size by.

        Returns
        -------
        SampleSize
            A new SampleSize object with the multiplied sample size.
        """
        return self.__mul__(factor)
