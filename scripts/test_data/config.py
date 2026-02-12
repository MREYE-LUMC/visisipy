from __future__ import annotations

from pathlib import Path

import numpy as np
from analysis_tests import (
    BaseAnalysisTest,
    FFTPSFTest,
    HuygensPSFTest,
    OPDMapTest,
    RayTraceTest,
    RefractionTest,
    ZernikeStandardCoefficientsTest,
)

from visisipy.backend import DEFAULT_BACKEND_SETTINGS
from visisipy.opticstudio.backend import OpticStudioSettings
from visisipy.optiland.backend import OptilandSettings

__all__ = [
    "DEFAULT_SAMPLING",
    "DEFAULT_WAVELENGTH",
    "OPTICSTUDIO_BACKEND_SETTINGS",
    "OPTILAND_BACKEND_SETTINGS",
    "TESTS",
    "TEST_DATA_DIR",
]

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "_data" / "analysis_results"

DEFAULT_WAVELENGTH = 0.543
DEFAULT_SAMPLING = 128

OPTICSTUDIO_BACKEND_SETTINGS = OpticStudioSettings(
    mode="standalone",
    ray_aiming="off",
    **DEFAULT_BACKEND_SETTINGS,
)
OPTILAND_BACKEND_SETTINGS = OptilandSettings(
    computation_backend="numpy",
    **DEFAULT_BACKEND_SETTINGS,
)

RAY_TRACE_COORDINATES = [(x, y) for x in np.linspace(-60, 60, 5) for y in np.linspace(-60, 60, 5)]

TESTS: dict[str, BaseAnalysisTest] = {
    "fft_psf_0_0": FFTPSFTest(coordinate=(0, 0), sampling=DEFAULT_SAMPLING, wavelength=DEFAULT_WAVELENGTH),
    "fft_psf_0_10": FFTPSFTest(coordinate=(0, 10), sampling=DEFAULT_SAMPLING, wavelength=DEFAULT_WAVELENGTH),
    "fft_psf_10_0": FFTPSFTest(coordinate=(10, 0), sampling=DEFAULT_SAMPLING, wavelength=DEFAULT_WAVELENGTH),
    "huygens_psf_0_0": HuygensPSFTest(
        coordinate=(0, 0),
        pupil_sampling=DEFAULT_SAMPLING,
        image_sampling=DEFAULT_SAMPLING,
        wavelength=DEFAULT_WAVELENGTH,
    ),
    "huygens_psf_0_10": HuygensPSFTest(
        coordinate=(0, 10),
        pupil_sampling=DEFAULT_SAMPLING,
        image_sampling=DEFAULT_SAMPLING,
        wavelength=DEFAULT_WAVELENGTH,
    ),
    "huygens_psf_10_0": HuygensPSFTest(
        coordinate=(10, 0),
        pupil_sampling=DEFAULT_SAMPLING,
        image_sampling=DEFAULT_SAMPLING,
        wavelength=DEFAULT_WAVELENGTH,
    ),
    "ray_trace_pupil_0_0": RayTraceTest(
        coordinates=RAY_TRACE_COORDINATES, pupil=(0, 0), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING
    ),
    "ray_trace_pupil_0_1": RayTraceTest(
        coordinates=RAY_TRACE_COORDINATES, pupil=(0, 1), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING
    ),
    "ray_trace_pupil_1_-1": RayTraceTest(
        coordinates=RAY_TRACE_COORDINATES, pupil=(1, -1), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING
    ),
    "refraction": RefractionTest(
        coordinates=[(x, y) for x in range(-10, 11, 5) for y in range(-10, 11, 5)],
        wavelength=DEFAULT_WAVELENGTH,
        sampling=DEFAULT_SAMPLING,
    ),
    "opd_map_0_0": OPDMapTest(
        coordinate=(0, 0), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, remove_tilt=False
    ),
    "opd_map_0_10": OPDMapTest(
        coordinate=(0, 10), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, remove_tilt=False
    ),
    "opd_map_10_0": OPDMapTest(
        coordinate=(10, 0), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, remove_tilt=False
    ),
    "opd_map_10_5": OPDMapTest(
        coordinate=(10, 5), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, remove_tilt=False
    ),
    "zernike_coefficients_0_0": ZernikeStandardCoefficientsTest(
        coordinate=(0, 0), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, maximum_term=45
    ),
    "zernike_coefficients_0_10": ZernikeStandardCoefficientsTest(
        coordinate=(0, 10), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, maximum_term=45
    ),
    "zernike_coefficients_10_0": ZernikeStandardCoefficientsTest(
        coordinate=(10, 0), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, maximum_term=45
    ),
    "zernike_coefficients_10_5": ZernikeStandardCoefficientsTest(
        coordinate=(10, 5), wavelength=DEFAULT_WAVELENGTH, sampling=DEFAULT_SAMPLING, maximum_term=45
    ),
}
