from __future__ import annotations

import pytest

from tests.helpers import build_args
from visisipy import EyeModel


class TestOPDMapAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling,remove_tilt",
        [
            (None, None, None, None, None),
            (None, None, "object_height", 64, False),
            ((0, 0), 0.550, "angle", 128, True),
            ((10, 5), 0.400, "object_height", 256, False),
            ((5.5, 5.5), 0.550, "angle", 512, True),
        ],
    )
    def test_opd_map(
        self, optiland_backend, optiland_analysis, field_coordinate, wavelength, field_type, sampling, remove_tilt
    ):
        optiland_backend.build_model(EyeModel(), object_distance=10 if field_type == "object_height" else float("inf"))

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            remove_tilt=remove_tilt,
            non_null_defaults={"field_type", "sampling", "remove_tilt"},
        )

        assert optiland_analysis.opd_map(**args)

    def test_opd_map_use_exit_pupil_shape_warning(self, optiland_backend, optiland_analysis):
        optiland_backend.build_model(EyeModel())

        with pytest.warns(UserWarning, match="Correcting for the exit pupil shape is not supported in Optiland."):
            optiland_analysis.opd_map(use_exit_pupil_shape=True)
