from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel
from visisipy.types import SampleSize

pytestmark = [pytest.mark.needs_opticstudio]


class TestZernikeStandardCoefficientsAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling,maximum_term",
        [
            (None, None, None, None, None),
            ((0, 0), 0.543, None, None, None),
            ((1, 1), 0.632, "angle", None, None),
            ((0.5, 0.5), 0.543, "object_height", 64, None),
            ((0, 0), 0.543, "angle", SampleSize(64), 45),
            ((1, 1), 0.632, "angle", "64x64", 100),
        ],
    )
    def test_zernike_standard_coefficients(
        self,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
        maximum_term,
        opticstudio_backend,
        opticstudio_analysis,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            maximum_term=maximum_term,
            non_null_defaults={"field_type", "sampling", "maximum_term"},
        )

        assert opticstudio_analysis.zernike_standard_coefficients(**args)

    @pytest.mark.parametrize("wavelengths", [(), (0.543,), (0.543, 0.632), (0.632, 0.543, 0.780)])
    def test_zernike_standard_coefficients_unit(self, wavelengths, opticstudio_backend, opticstudio_analysis):
        opticstudio_backend.set_wavelengths(wavelengths or (0.543,))

        wavelength = wavelengths[0] if wavelengths else None

        opticstudio_backend.build_model(EyeModel())

        coefficients_waves, _ = opticstudio_analysis.zernike_standard_coefficients(wavelength=wavelength, unit="waves")
        coefficients_microns, _ = opticstudio_analysis.zernike_standard_coefficients(
            wavelength=wavelength, unit="microns"
        )

        # If no wavelength is specified, the analysis should use the first wavelength in the backend
        if wavelength is None:
            wavelength = opticstudio_backend.get_wavelengths()[0]

        assert not np.allclose(
            list(coefficients_waves.values()), 0
        )  # Sanity check; the test is meaningless if all coefficients are zero
        assert all(
            coefficients_microns[index] == pytest.approx(coefficients_waves[index] * wavelength)
            for index in coefficients_waves
        )

    def test_zernike_standard_coefficients_maximum_term(self, opticstudio_analysis):
        coefficients, _ = opticstudio_analysis.zernike_standard_coefficients(maximum_term=20)

        assert coefficients[21] == 0
