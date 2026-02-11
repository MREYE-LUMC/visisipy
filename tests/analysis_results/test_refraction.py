from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import visisipy

if TYPE_CHECKING:
    from pandas import DataFrame, Series

    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


def get_result(df: DataFrame, field_coordinate: tuple[float, float]) -> Series:
    rows = df.query("field_x == @field_coordinate[0] and field_y == @field_coordinate[1]")

    if (n := len(rows)) != 1:
        raise ValueError(f"Expected exactly one row for field coordinate {field_coordinate}, but got {n}")

    return rows.iloc[0]


@pytest.mark.test_data("refraction")
@pytest.mark.parametrize("field", [(x, y) for x in range(-10, 11, 5) for y in range(-10, 11, 5)])
def test_refraction(
    field, result_test_model: EyeModel, configure_backend: type[BaseBackend], expected_result: DataFrame
):
    result = visisipy.analysis.refraction(
        model=result_test_model,
        field_coordinate=field,
        sampling=128,
        wavelength=0.543,
        backend=configure_backend,
    )
    expected = get_result(expected_result, field)

    assert pytest.approx(expected.M, abs=1e-6) == result.M
    assert pytest.approx(expected.J0, abs=1e-6) == result.J0
    assert pytest.approx(expected.J45, abs=1e-6) == result.J45
