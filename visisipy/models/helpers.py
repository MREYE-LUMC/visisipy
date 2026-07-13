"""Helper functions for eye models."""

from __future__ import annotations

from typing import NamedTuple, TypeVar

import numpy as np

T = TypeVar("T", bound=type)


def _collect_subclasses(cls: T, registry: dict[str, T]) -> None:
    registry[cls.__name__] = cls
    for subclass in cls.__subclasses__():
        _collect_subclasses(subclass, registry)


class _ConicCurvature(NamedTuple):
    radius: float
    asphericity: float


def radii_to_curvature(radius_a: float, radius_b: float) -> _ConicCurvature:
    r"""Convert conic radii to curvature and asphericity.

    The curvature and asphericity are calculated from the conic radii using the following formulas:

    .. math::
        R = \frac{R_a^2}{R_b} \\
        K = \left(\frac{R_a}{R_b}\right)^2 - 1

    Parameters
    ----------
    radius_a : float
        The first conic radius.
    radius_b : float
        The second conic radius.

    Returns
    -------
    _ConicCurvature
        A named tuple containing the radius of curvature and asphericity.
    """
    radius = radius_a**2 / radius_b
    asphericity = (radius_a / radius_b) ** 2 - 1

    return _ConicCurvature(radius=radius, asphericity=asphericity)


def curvature_to_radii(radius: float, asphericity: float) -> tuple[float, float]:
    r"""Convert a radius of curvature and asphericity to conic radii.

    The conic radii are calculated from the curvature and asphericity using the following formulas:

    .. math::
        R_a = \frac{R}{\sqrt{ K + 1 }} \\
        R_b = \frac{R}{K + 1}

    Parameters
    ----------
    radius : float
        The radius of curvature.
    asphericity : float
        The asphericity.

    Returns
    -------
    float
        The first conic radius corresponding to the given curvature.
    float
        The second conic radius corresponding to the given curvature.
    """
    radius_a = radius / np.sqrt(asphericity + 1)
    radius_b = radius / (asphericity + 1)

    return radius_a, radius_b
