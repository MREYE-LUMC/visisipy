import numpy as np
import pytest

from visisipy import EyeGeometry, NavarroGeometry


@pytest.fixture
def geometry_parameters():
    return dict(
        axial_length=20,
        cornea_thickness=0.5,
        anterior_chamber_depth=3,
        lens_thickness=4,
        cornea_front_curvature=7,
        cornea_front_asphericity=0,
        cornea_back_curvature=6,
        cornea_back_asphericity=0,
        lens_front_curvature=10,
        lens_front_asphericity=0,
        lens_back_curvature=-6,
        lens_back_asphericity=0,
        retina_curvature=-12,
        retina_asphericity=0,
        iris_radius=1,
    )


@pytest.mark.parametrize("geometry", [EyeGeometry, NavarroGeometry])
@pytest.mark.parametrize("asphericity", [-1, -1.5])
def test_non_ellipsoid_retina_raises_valueerror(geometry, asphericity, geometry_parameters):
    geometry_parameters.update(retina_asphericity=asphericity)

    with pytest.raises(ValueError):
        geometry(**geometry_parameters)


def test_vitreous_thickness(geometry_parameters):
    vitreous_thickness = geometry_parameters["axial_length"] - (
        geometry_parameters["cornea_thickness"]
        + geometry_parameters["anterior_chamber_depth"]
        + geometry_parameters["lens_thickness"]
    )

    geometry = EyeGeometry(**geometry_parameters)

    assert geometry.vitreous_thickness == vitreous_thickness


@pytest.mark.parametrize(
    "curvature,asphericity,radial_axis,axial_axis",
    [
        (10, 0, 10, 10),  # Circle
        (10, -0.5, 14.14213562, 20),  # Prolate ellipse
        (10, 0.5, 8.164965809, 6.666666667),  # Oblate ellipse
        (-10, 0, 10, 10),  # Circle
        (-10, -0.5, 14.14213562, 20),  # Prolate ellipse
        (-10, 0.5, 8.164965809, 6.666666667),  # Oblate ellipse
    ],
)
def test_retinal_half_axes(geometry_parameters, curvature, asphericity, radial_axis, axial_axis):
    geometry_parameters.update(retina_curvature=curvature, retina_asphericity=asphericity)
    geometry = EyeGeometry(**geometry_parameters)

    assert np.isclose(geometry.retina_axial_half_axis, axial_axis)
    assert np.isclose(geometry.retina_radial_half_axis, radial_axis)
