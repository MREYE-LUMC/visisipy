from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from tests.helpers import build_args
from visisipy import EyeModel
from visisipy.opticstudio.analysis.psf import _get_huygens_psf_extent, _opticstudio_batch_raytrace

pytestmark = [pytest.mark.needs_opticstudio]


class TestFFTPSFAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling",
        [
            (None, None, None, None),
            (None, None, "object_height", 64),
            ((0, 0), 0.550, "angle", 128),
            ((10, 5), 0.400, "object_height", 256),
            ((5.5, 5.5), 0.550, "angle", 512),
        ],
    )
    def test_fft_psf(
        self,
        opticstudio_backend,
        opticstudio_analysis,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            non_null_defaults={"field_type", "sampling"},
        )

        assert opticstudio_analysis.fft_psf(**args)


class TestHuygensPSFAnalysis:
    @pytest.mark.parametrize(
        "px,py,expectation",
        [
            (np.linspace(0, 1, 10), np.linspace(0, 1, 10), does_not_raise()),
            (
                np.linspace(0, 1, 10),
                np.linspace(0, 1, 20),
                pytest.raises(ValueError, match="p_x and p_y must have the same length"),
            ),
        ],
    )
    def test_batch_raytrace(self, opticstudio_backend, px, py, expectation):
        oss = opticstudio_backend.get_oss()
        oss.new()

        assert oss.Tools.CurrentTool is None

        with expectation:
            raytrace_results = _opticstudio_batch_raytrace(
                oss,
                wavelength_number=1,
                h_x=0,
                h_y=0,
                p_x=px,
                p_y=py,
            )

            assert oss.Tools.CurrentTool is None

            assert len(raytrace_results) == len(px)
            assert all(result.x == 0 and result.y == 0 for result in raytrace_results)

    def test_batch_raytrace_closes_current_tool(self, opticstudio_backend):
        oss = opticstudio_backend.get_oss()
        oss.Tools.OpenLocalOptimization()

        _opticstudio_batch_raytrace(oss, wavelength_number=1, h_x=0, h_y=0, p_x=[0], p_y=[0])

        assert oss.Tools.CurrentTool is None

    def test_get_huygens_psf_extent(self, opticstudio_backend):
        oss = opticstudio_backend.get_oss()

        oss.new()

        oss.SystemData.Aperture.ApertureValue = 1

        lens_front = oss.LDE.InsertNewSurfaceAt(2)
        lens_front.Thickness = 10
        lens_front.Radius = 10
        lens_front.Material = "BK7"

        lens_back = oss.LDE.InsertNewSurfaceAt(3)
        lens_back.Thickness = 10
        lens_back.Radius = -10

        assert _get_huygens_psf_extent(oss) == pytest.approx(203.66247163766525)

    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,pupil_sampling,image_sampling",
        [
            (None, None, None, None, None),
            (None, None, "object_height", 32, 32),
            ((0, 0), 0.550, "angle", 64, 128),
            ((10, 5), 0.400, "object_height", 256, 128),
            ((5.5, 5.5), 0.550, "angle", 64, 64),
        ],
    )
    def test_fft_psf(
        self,
        opticstudio_backend,
        opticstudio_analysis,
        field_coordinate,
        wavelength,
        field_type,
        pupil_sampling,
        image_sampling,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            pupil_sampling=pupil_sampling,
            image_sampling=image_sampling,
            non_null_defaults={"field_type", "pupil_sampling", "image_sampling"},
        )

        assert opticstudio_analysis.huygens_psf(**args)


class TestStrehlRatioAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength,field_type,sampling",
        [
            (None, None, None, None),
            (None, None, "object_height", 64),
            ((0, 0), 0.550, "angle", 128),
            ((10, 5), 0.400, "object_height", 256),
            ((5.5, 5.5), 0.550, "angle", 64),
        ],
    )
    def test_strehl_ratio(
        self,
        opticstudio_backend,
        opticstudio_analysis,
        field_coordinate,
        wavelength,
        field_type,
        sampling,
    ):
        opticstudio_backend.build_model(
            EyeModel(), object_distance=10 if field_type == "object_height" else float("inf")
        )

        args = build_args(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            sampling=sampling,
            non_null_defaults={"field_type", "sampling"},
        )

        assert opticstudio_analysis.strehl_ratio(**args, psf_type="huygens")

    @pytest.mark.parametrize(
        "psf_type,expectation",
        [
            (
                "fft",
                pytest.raises(
                    NotImplementedError,
                    match="OpticStudio does not support obtaining the Strehl ratio from the FFT PSF",
                ),
            ),
            ("huygens", does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    NotImplementedError, match="PSF type 'invalid' is not implemented. Only 'huygens' is supported."
                ),
            ),
        ],
    )
    def test_psf_type(self, opticstudio_backend, opticstudio_analysis, psf_type, expectation):
        opticstudio_backend.build_model(EyeModel(), object_distance=float("inf"))

        with expectation:
            opticstudio_analysis.strehl_ratio(psf_type=psf_type)
