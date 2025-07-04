from __future__ import annotations

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

    def test_zernike_standard_coefficients_maximum_term(self, opticstudio_analysis):
        coefficients, _ = opticstudio_analysis.zernike_standard_coefficients(maximum_term=20)

        assert coefficients[21] == 0
