from contextlib import nullcontext as does_not_raise

import pytest

pytestmark = [pytest.mark.needs_opticstudio]


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
                [0.543],
                "object_height",
                (1, 0),
                does_not_raise(),
                marks=pytest.mark.xfail(
                    reason="ZOSPy cannot parse ray trace results when field_type is object_height"
                ),
            ),
            (
                [(0, 0), (1, 1)],
                [0.543],
                "angle",
                (2, 0),
                pytest.raises(
                    ValueError, match="Pupil coordinates must be between -1 and 1"
                ),
            ),
            (
                [(0, 0), (1, 1)],
                [0.543],
                "angle",
                (0, 2),
                pytest.raises(
                    ValueError, match="Pupil coordinates must be between -1 and 1"
                ),
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
        opticstudio_analysis,
    ):
        args = {
            "coordinates": coordinates,
            "wavelengths": wavelengths,
        }

        if field_type is not None:
            args["field_type"] = field_type
        if pupil is not None:
            args["pupil"] = pupil

        with expectation:
            assert opticstudio_analysis.raytrace(**args)

    def test_raytrace_dataframe_structure(self, opticstudio_analysis):
        coordinates = [(0, 0), (1, 1)]

        result, _ = opticstudio_analysis.raytrace(coordinates)
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
