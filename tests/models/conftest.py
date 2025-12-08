from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from visisipy.models.geometry import EyeGeometry, StandardSurface, Stop

if TYPE_CHECKING:
    from visisipy.models.factory import GeometryParameters


@pytest.fixture
def example_geometry_parameters() -> GeometryParameters:
    return {
        "axial_length": 20,
        "cornea_thickness": 0.5,
        "anterior_chamber_depth": 3,
        "lens_thickness": 4,
        "cornea_front_radius": 7,
        "cornea_front_asphericity": 0,
        "cornea_back_radius": 6,
        "cornea_back_asphericity": 0,
        "lens_front_radius": 10,
        "lens_front_asphericity": 0,
        "lens_back_radius": -6,
        "lens_back_asphericity": 0,
        "retina_radius": -12,
        "retina_asphericity": 0,
        "pupil_radius": 1.0,
        "pupil_lens_distance": 0.4,
    }


@pytest.fixture
def example_geometry(example_geometry_parameters):
    return EyeGeometry(
        cornea_front=StandardSurface(
            radius=example_geometry_parameters["cornea_front_radius"],
            asphericity=example_geometry_parameters["cornea_front_asphericity"],
            thickness=example_geometry_parameters["cornea_thickness"],
        ),
        cornea_back=StandardSurface(
            radius=example_geometry_parameters["cornea_back_radius"],
            asphericity=example_geometry_parameters["cornea_back_asphericity"],
            thickness=example_geometry_parameters["anterior_chamber_depth"],
        ),
        pupil=Stop(
            semi_diameter=example_geometry_parameters["pupil_radius"],
            thickness=example_geometry_parameters["pupil_lens_distance"],
        ),
        lens_front=StandardSurface(
            radius=example_geometry_parameters["lens_front_radius"],
            asphericity=example_geometry_parameters["lens_front_asphericity"],
            thickness=example_geometry_parameters["lens_thickness"],
        ),
        lens_back=StandardSurface(
            radius=example_geometry_parameters["lens_back_radius"],
            asphericity=example_geometry_parameters["lens_back_asphericity"],
            thickness=round(
                example_geometry_parameters["axial_length"]
                - example_geometry_parameters["cornea_thickness"]
                - example_geometry_parameters["anterior_chamber_depth"]
                - example_geometry_parameters["pupil_lens_distance"]
                - example_geometry_parameters["lens_thickness"],
                7,
            ),
        ),
        retina=StandardSurface(
            radius=example_geometry_parameters["retina_radius"],
            asphericity=example_geometry_parameters["retina_asphericity"],
        ),
    )
