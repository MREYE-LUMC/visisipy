from __future__ import annotations

import numpy as np
import pytest
from optiland.samples import NavarroWideAngleEye
from optiland.wavefront import OPD

from tests.helpers import build_args
from visisipy import EyeModel
from visisipy.optiland.analysis.wavefront import generate_opd_map


class TestGenerateOPDMap:
    @pytest.mark.parametrize("field", [(0, 0), (1 / np.sqrt(2), 1 / np.sqrt(2)), (0, 1)])
    def test_generate_opd_map(self, field):
        model = NavarroWideAngleEye()
        wavelength = 0.543
        sampling = 32

        opd = OPD(model, field=field, wavelength=wavelength, num_rays=sampling, distribution="uniform")

        # Generate OPD map using Optiland's interpolation method
        data_optiland = opd.generate_opd_map(num_points=sampling)

        # Generate OPD map using the index-based method
        data_fast = generate_opd_map(
            opd.get_data(field=field, wl=wavelength), distribution=opd.distribution, sampling=sampling
        )

        np.testing.assert_allclose(data_fast["z"], data_optiland["z"])
        np.testing.assert_array_equal(data_fast["x"], data_optiland["x"])
        np.testing.assert_array_equal(data_fast["y"], data_optiland["y"])

    def test_generate_opd_map_random(self):
        model = NavarroWideAngleEye()
        field = (0, 0)
        wavelength = 0.543
        sampling = 32

        opd = OPD(model, field=field, wavelength=wavelength, num_rays=sampling, distribution="uniform")

        # Replace the OPD data with random data
        np.random.seed(0)
        opd.data[field, wavelength].opd = np.random.randint(-100, 100, size=opd.data[field, wavelength].opd.shape)

        # Generate OPD map using Optiland's interpolation method
        data_optiland = opd.generate_opd_map(num_points=sampling)

        # Generate OPD map using the index-based method
        data_fast = generate_opd_map(
            opd.get_data(field=field, wl=wavelength), distribution=opd.distribution, sampling=sampling
        )

        np.testing.assert_allclose(data_fast["z"], data_optiland["z"])


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
