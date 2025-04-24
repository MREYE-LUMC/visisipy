"""Optical analyses for eye models.

This module provides access to various optical analyses for eye models.
The simulations themselves are implemented for each backend.
This module only provides a common interface for the analyses.

Analysis are functions with the following signature:

    analysis(model, [parameters], return_raw_result, backend)

The first parameter is the eye model to be used in the analysis. This parameter is optional;
if no model is provided, the model that is currently defined in the backend will be used.
Setting the return_raw_result parameter to True will return the raw analysis result from the backend.
The backend parameter is also optional; if no backend is provided, the current or default backend will be used.
"""

from __future__ import annotations

from visisipy.analysis.cardinal_points import cardinal_points
from visisipy.analysis.raytracing import raytrace
from visisipy.analysis.refraction import refraction
from visisipy.analysis.zernike_standard_coefficients import (
    zernike_standard_coefficients,
)

__all__ = ("cardinal_points", "raytrace", "refraction", "zernike_standard_coefficients")
