from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from tests.helpers import build_args
from visisipy.models import EyeModel


class TestRayTraceAnalysis:
    @pytest.mark.parametrize(
        "coordinates,wavelengths,field_type,pupil,expectation",
        [
            (None, None, None, None, does_not_raise()),
            ([(0, 0), (1, 1)], None, None, None, does_not_raise()),
            ([(0, 0), (1, 1)], [0.543], None, None, does_not_raise()),
            ([(0, 0), (1, 1)], [0.543], "angle", None, does_not_raise()),
            pytest.param(
                [(0, 0), (1, 1)],
                [0.543, 0.600],
                "object_height",
                (1, 0),
                does_not_raise(),
            ),
            (
                [(0, 0), (1, 1)],
                [0.543],
                "angle",
                (2, 0),
                pytest.raises(ValueError, match="Pupil coordinates must be between -1 and 1"),
            ),
            (
                [(0, 0), (1, 1)],
                [0.543, 0.600],
                "angle",
                (0, 2),
                pytest.raises(ValueError, match="Pupil coordinates must be between -1 and 1"),
            ),
        ],
    )
    def test_raytrace(
        self,
        coordinates,
        wavelengths,
        field_type,
        pupil,
        expectation,
        optiland_backend,
        optiland_analysis,
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        args = build_args(
            non_null_defaults={"field_type", "pupil"},
            coordinates=coordinates,
            wavelengths=wavelengths,
            field_type=field_type,
            pupil=pupil,
        )

        with expectation:
            assert optiland_analysis.raytrace(**args)

    def test_raytrace_dataframe_structure(self, optiland_backend, optiland_analysis):
        optiland_backend.build_model(EyeModel())

        coordinates = [(0, 0), (1, 1)]

        result, _ = optiland_analysis.raytrace(coordinates)
        expected_columns = {
            "index",
            "x",
            "y",
            "z",
            "field",
            "wavelength",
            "surface",
            "comment",
        }

        assert set(result.columns) == expected_columns
        assert set(result.field.unique()) == set(coordinates)
