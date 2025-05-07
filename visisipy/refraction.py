"""Utilities for handling refraction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = (
    "FourierPowerVectorRefraction",
    "PolarPowerVectorRefraction",
    "SpheroCylindricalRefraction",
)


@dataclass
class FourierPowerVectorRefraction:
    """Ocular refraction in Fourier power vector form.

    This is the default representation of refractions in visisipy. This class contains several methods to convert the
    refraction to other forms, using the conversions defined in [1].

    .. [1] Thibos LN, Wheeler W, Horner D. Power vectors: an application of Fourier analysis to the description and
           statistical analysis of refractive error. Optometry and vision science: official publication of the American
           Academy of Optometry. 1997 Jun;74(6):367-75.

    Attributes
    ----------
    M : float
        The power of the refraction.
    J0 : float
        The Jackson cross-cylinder power at 0°.
    J45 : float
        The Jackson cross-cylinder power at 45°.

    Methods
    -------
    to_polar_power_vectors() -> PolarPowerVectorRefraction
        Converts the refraction to polar power vector form.
    to_sphero_cylindrical(cylinder_form: Literal["positive", "negative"] = "negative") -> SpheroCylindricalRefraction
        Converts the refraction to sphero-cylindrical form.
    """

    M: float
    J0: float
    J45: float

    def to_polar_power_vectors(self) -> PolarPowerVectorRefraction:
        """Converts the refraction to polar power vector form.

        Returns
        -------
        PolarPowerVectorRefraction
            The refraction in polar power vector form.
        """
        return PolarPowerVectorRefraction(
            M=self.M,
            J=np.sqrt(self.J0**2 + self.J45**2),
            axis=np.rad2deg(np.arctan2(self.J45, self.J0) / 2),
        )

    def to_sphero_cylindrical(
        self, cylinder_form: Literal["positive", "negative"] = "negative"
    ) -> SpheroCylindricalRefraction:
        """Converts the refraction to sphero-cylindrical form.

        Parameters
        ----------
        cylinder_form : str, optional
            Indicates if the cylinder should be positive or negative. Defaults to "negative".

        Returns
        -------
        SpheroCylindricalRefraction
            The refraction in sphero-cylindrical form.
        """
        if cylinder_form not in {"positive", "negative"}:
            raise ValueError("cylinder_form must be either 'positive' or 'negative'.")

        sphero_cylinder = SpheroCylindricalRefraction(
            sphere=self.M + np.sqrt(self.J0**2 + self.J45**2),
            cylinder=-2 * np.sqrt(self.J0**2 + self.J45**2),
            axis=np.rad2deg(np.arctan2(self.J45, self.J0) / 2),
        )

        return sphero_cylinder.convert_cylinder_form(cylinder_form)


@dataclass
class PolarPowerVectorRefraction:
    """Ocular refraction in polar power vector form.

    Attributes
    ----------
    M : float
        The power of the refraction.
    J : float
        The Jackson cross-cylinder power.
    axis : float
        The axis of the Jackson cross-cylinder power.
    """

    M: float
    J: float
    axis: float


@dataclass
class SpheroCylindricalRefraction:
    """Ocular refraction in sphero-cylindrical form.

    Attributes
    ----------
    sphere : float
        The spherical component of the refraction.
    cylinder : float
        The cylindrical component of the refraction.
    axis : float
        The axis of the cylindrical component of the refraction.

    Methods
    -------
    has_positive_cylinder() -> bool
        Returns `True` if the cylinder is positive or `NaN`, `False` otherwise.
    has_negative_cylinder() -> bool
        Returns `True` if the cylinder is negative or `NaN`, `False` otherwise.
    convert_cylinder_form(to: Literal["positive", "negative"]) -> SpheroCylindricalRefraction
        Converts from positive cylinder refraction to negative cylinder refraction and vice-versa.
    """

    sphere: float
    cylinder: float
    axis: float

    @property
    def has_positive_cylinder(self) -> bool:
        """Returns `True` if the cylinder is positive or `NaN`, `False` otherwise."""
        return np.isnan(self.cylinder) or self.cylinder >= 0

    @property
    def has_negative_cylinder(self) -> bool:
        """Returns `True` if the cylinder is negative or `NaN`, `False` otherwise."""
        return np.isnan(self.cylinder) or self.cylinder < 0

    def _convert_cylinder_form(self) -> SpheroCylindricalRefraction:
        return SpheroCylindricalRefraction(
            sphere=self.sphere + self.cylinder,
            cylinder=-self.cylinder,
            axis=(self.axis + 90) % 180,
        )

    def convert_cylinder_form(self, to: Literal["positive", "negative"]) -> SpheroCylindricalRefraction:
        """Converts from positive cylinder refraction to negative cylinder refraction and vice-versa.

        Parameters
        ----------
        to : str
            Indicates if the conversion should be done towards negative cylinder form ("negative") or positive cylinder
            form ("positive").

        Returns
        -------
        SpheroCylindricalRefraction, optional
            With inplace set to True, no return is given as the current class is updated. With inplace set to False, a
            copy of the current SpheroCylindricalRefraction instance is returned, but in the specified cylinder format.

        Raises
        ------
        ValueError
            When the parameter "to" is not set to 'positive' or 'negative'.
        """

        if to not in {"negative", "positive"}:
            raise ValueError('"to" should be either "negative" or "positive"')

        if (to == "negative" and self.has_negative_cylinder) or (to == "positive" and self.has_positive_cylinder):
            # Conversion not needed
            return self

        return self._convert_cylinder_form()
