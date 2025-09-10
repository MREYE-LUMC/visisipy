from __future__ import annotations

import numpy as np
import pytest
from pandas import DataFrame

from tests.helpers import build_args
from visisipy import EyeModel
from visisipy.opticstudio.analysis.wavefront import _pad_index, _pad_opd_map


class TestOPDMapAnalysis:
    def test_pad_index(self):
        index = np.arange(0, 5, 0.123)
        padded_index = _pad_index(index)

        assert len(padded_index) == len(index) + 1
        assert padded_index[0] == -0.123

    def test_pad_opd_map(self):
        opd_map = DataFrame(
            np.arange(9).reshape(3, 3),
            index=[-1, 0, 1],
            columns=[-1, 0, 1],
            dtype=float,
        )

        padded_opd_map = _pad_opd_map(opd_map)

        assert padded_opd_map.shape == (4, 4)
        assert padded_opd_map.index[0] == -2
        assert padded_opd_map.columns[0] == -2
        assert np.all(np.isnan(padded_opd_map.to_numpy()[0, :]))
        assert np.all(np.isnan(padded_opd_map.to_numpy()[:, 0]))

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
        self, opticstudio_backend, opticstudio_analysis, field_coordinate, wavelength, field_type, sampling, remove_tilt
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            remove_tilt=remove_tilt,
            non_null_defaults={"field_type", "sampling", "remove_tilt"},
        )

        assert opticstudio_analysis.opd_map(**args)
