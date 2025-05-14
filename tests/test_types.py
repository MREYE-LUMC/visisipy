from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.types import SampleSize


class TestSampleSize:
    @pytest.mark.parametrize(
        "input_,value,expectation",
        [
            (32, 32, does_not_raise()),
            ("64x64", 64, does_not_raise()),
            ("512X512", 512, does_not_raise()),
            (SampleSize(128), 128, does_not_raise()),
            ("123", 123, pytest.raises(ValueError, match="Invalid sample size format: 123")),
            ("123x456", 123, pytest.raises(ValueError, match="Invalid sample size format: 123x456")),
        ],
    )
    def test_samplesize(self, input_, value, expectation):
        with expectation:
            sampling = SampleSize(input_)
            assert sampling.sampling == value

    def test_int_conversion(self):
        assert int(SampleSize(32)) == 32

    def test_str_conversion(self):
        assert str(SampleSize(32)) == "32x32"
