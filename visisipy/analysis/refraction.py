from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

from visisipy.analysis.base import analysis
from visisipy._backend import get_backend

if TYPE_CHECKING:
    from visisipy import EyeModel


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
        if cylinder_form not in ("positive", "negative"):
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

    def convert_cylinder_form(
        self, to: Literal["positive", "negative"]
    ) -> SpheroCylindricalRefraction:
        """Converts from positive cylinder refraction to negative cylinder refraction and vice-versa.

        Parameters
        ----------
        to: str
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

        if to not in ("negative", "positive"):
            raise ValueError('"to" should be either "negative" or "positive"')

        if (to == "negative" and self.has_negative_cylinder) or (
            to == "positive" and self.has_positive_cylinder
        ):
            # Conversion not needed
            return self
        else:
            return self._convert_cylinder_form()


@analysis
def refraction(
    model: EyeModel | None,
    use_higher_order_aberrations: bool = True,
    field_coordinate: tuple[float, float] | None = None,
    wavelength: float | None = None,
    pupil_diameter: float | None = None,
    field_type: Literal["angle", "object_height"] = "angle",
    *,
    return_raw_result: bool = False,
) -> FourierPowerVectorRefraction | tuple[FourierPowerVectorRefraction, Any]:
    """Calculates the ocular refraction.

    The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power vector form.

    Parameters
    ----------
    use_higher_order_aberrations : bool, optional
        If `True`, higher-order aberrations are used in the calculation. Defaults to `True`.
    field_coordinate : tuple[float, float], optional
        The field coordinate for the Zernike calculation. When `None`, the default field configured in the backend is
        used. Defaults to `None`.
    wavelength : float, optional
        The wavelength for the Zernike calculation. When `None`, the default wavelength configured in the backend is
        used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The type of field to be used when setting the field coordinate. This parameter is only used when
        `field_coordinate` is specified. Defaults to "angle".
    pupil_diameter : float, optional
        The diameter of the pupil for the refraction calculation. Defaults to the pupil diameter configured in the
        model.
    return_raw_result : bool, optional
        Return the raw analysis result from the backend. Defaults to `False`.

    Returns
    -------
     FourierPowerVectorRefraction
          The ocular refraction in Fourier power vector form.
    Any
        The raw analysis result from the backend.
    """
    return get_backend().analysis.refraction(
        use_higher_order_aberrations=use_higher_order_aberrations,
        field_coordinate=field_coordinate,
        wavelength=wavelength,
        pupil_diameter=pupil_diameter,
        field_type=field_type,
    )
