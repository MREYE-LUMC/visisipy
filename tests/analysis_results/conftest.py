from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pandas import DataFrame, read_csv

from visisipy.models import EyeGeometry, EyeModel
from visisipy.models.geometry import BiconicSurface, StandardSurface, Stop
from visisipy.models.materials import NavarroMaterials543

if TYPE_CHECKING:
    from visisipy.backend import BaseBackend


@pytest.fixture
def result_test_model() -> EyeModel:
    """Eye model for analysis results tests."""
    materials = NavarroMaterials543()
    geometry = EyeGeometry(
        cornea_front=BiconicSurface(
            radius=7.6967, asphericity=-0.2304, thickness=0.5615, radius_x=7.9487, asphericity_x=-0.2304
        ),
        cornea_back=StandardSurface(radius=6.2343, asphericity=-0.1444, thickness=3.345),
        pupil=Stop(semi_diameter=1.0),
        lens_front=StandardSurface(radius=10.2, asphericity=-3.1316, thickness=3.17),
        lens_back=StandardSurface(radius=-5.4537, asphericity=0, thickness=17.2285),
        retina=StandardSurface(radius=-12.5000, asphericity=0.033),
    )
    return EyeModel(geometry=geometry, materials=materials)


@pytest.fixture
def expected_result(request: pytest.FixtureRequest, configure_backend: type[BaseBackend]) -> DataFrame:
    """Expected results for the analysis tests."""
    test_data_marker = request.node.get_closest_marker("test_data")

    if test_data_marker is None:
        raise ValueError("No test_data marker found on the test function.")

    test_name = test_data_marker.args[0]
    header_type = test_data_marker.kwargs.get("header_type")
    index_type = test_data_marker.kwargs.get("index_type")
    backend_name = str(configure_backend.type)

    data_file = request.config.rootpath / "tests" / "_data" / "analysis_results" / f"{test_name}_{backend_name}.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Expected results file not found: {data_file}")

    csv = read_csv(data_file, index_col=0)

    if header_type:
        csv.columns = csv.columns.astype(header_type)
    if index_type:
        csv.index = csv.index.astype(index_type)

    return csv
