from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from visisipy.models import EyeGeometry
from visisipy.models.geometry import (
    BaseZernikeStandardSurface,
    BiconicSurface,
    StandardSurface,
    Stop,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)


class TestStandardSurface:
    @pytest.mark.parametrize(
        "radius,asphericity,z_radius,y_radius",
        [
            (12, 0, 12, 12),
            (12, 1, 6, 12 / np.sqrt(2)),
            (12, -0.5, 24, 12 / np.sqrt(0.5)),
        ],
    )
    def test_ellipsoid_radii(self, radius, asphericity, z_radius, y_radius):
        surface = StandardSurface(radius=radius, asphericity=asphericity)

        assert surface.ellipsoid_radii == pytest.approx((z_radius, y_radius, y_radius))
        assert surface.ellipsoid_radii.z == pytest.approx(z_radius)
        assert surface.ellipsoid_radii.y == pytest.approx(y_radius)
        assert surface.ellipsoid_radii.x == pytest.approx(y_radius)
        assert surface.ellipsoid_radii.anterior_posterior == pytest.approx(z_radius)
        assert surface.ellipsoid_radii.inferior_superior == pytest.approx(y_radius)
        assert surface.ellipsoid_radii.left_right == pytest.approx(y_radius)

    @pytest.mark.parametrize("asphericity", [-1, -1.5])
    def test_ellipsoid_radii_raises_notimplementederror(self, asphericity):
        surface = StandardSurface(radius=12, asphericity=asphericity)

        with pytest.raises(NotImplementedError):
            _ = surface.ellipsoid_radii


class TestStop:
    def test_init_stop(self):
        stop = Stop()

        assert stop.is_stop
        assert stop.semi_diameter > 0


class TestBiconicSurface:
    @pytest.mark.parametrize(
        "radius_y,asphericity_y,radius_x,asphericity_x,z_radius,y_radius,x_radius",
        [
            (12, 0, 12, 0, 12, 12, 12),  # Sphere
            (12, 1, 12, 1, 6, 12 / np.sqrt(2), 12 / np.sqrt(2)),  # Rotationally symmetric ellipsoid
            (144 / 14, 144 / 196 - 1, 100 / 14, 100 / 196 - 1, 14, 12, 10),  # Astigmatic ellipsoid
        ],
    )
    def test_ellipsoid_radii(self, radius_y, asphericity_y, radius_x, asphericity_x, z_radius, y_radius, x_radius):
        surface = BiconicSurface(
            radius=radius_y, asphericity=asphericity_y, radius_x=radius_x, asphericity_x=asphericity_x
        )

        assert surface.ellipsoid_radii == pytest.approx((z_radius, y_radius, x_radius))
        assert surface.ellipsoid_radii.z == pytest.approx(z_radius)
        assert surface.ellipsoid_radii.y == pytest.approx(y_radius)
        assert surface.ellipsoid_radii.x == pytest.approx(x_radius)

    @pytest.mark.parametrize(
        "parameter,value", [("asphericity", -1), ("asphericity_x", -1), ("asphericity", -1.5), ("asphericity_x", -1.5)]
    )
    def test_non_ellipsoid_asphericity_raises_notimplementederror(self, parameter, value):
        surface = BiconicSurface(radius=12, radius_x=12, **{parameter: value})

        with pytest.raises(
            NotImplementedError, match=re.escape("Half axes are only defined for ellipsoids (asphericity > -1)")
        ):
            _ = surface.ellipsoid_radii

    def test_non_ellipsoid_raises_notimplementederror(self):
        surface = BiconicSurface(radius=13, asphericity=1, radius_x=12, asphericity_x=0)

        with pytest.raises(
            NotImplementedError,
            match=re.escape("Half axes are only defined for ellipsoids. This biconic surface is not an ellipsoid"),
        ):
            _ = surface.ellipsoid_radii


class TestZernikeSurfaces:
    def test_base_zernike_standard_surface_raises_typeerror(self):
        with pytest.raises(TypeError, match="Cannot instantiate abstract class BaseZernikeStandardSurface."):
            BaseZernikeStandardSurface(zernike_coefficients={}, maximum_term=None)

    @pytest.mark.parametrize("surface_type", [ZernikeStandardSagSurface, ZernikeStandardPhaseSurface])
    @pytest.mark.parametrize(
        "zernike_coefficients, maximum_term, expected_maximum_term, expectation",
        [
            ({}, None, 0, does_not_raise()),
            ({1: 0.1, 2: 0.2}, None, 2, does_not_raise()),
            ({1: 0.1, 2: 0.2}, 4, 4, does_not_raise()),
            ({}, 5, 5, does_not_raise()),
            (
                {1: 0.1, 2: 0.2},
                1,
                1,
                pytest.raises(
                    ValueError, match="The Zernike coefficients contain terms that are greater than the maximum term."
                ),
            ),
        ],
    )
    def test_init(self, surface_type, zernike_coefficients, maximum_term, expected_maximum_term, expectation):
        with expectation:
            surface = surface_type(
                zernike_coefficients=zernike_coefficients,
                maximum_term=maximum_term,
            )

            assert surface.zernike_coefficients == zernike_coefficients
            assert surface.maximum_term == expected_maximum_term


class TestEyeGeometry:
    def test_init_geometry(self):
        assert EyeGeometry(
            cornea_front=StandardSurface(),
            cornea_back=StandardSurface(),
            pupil=Stop(),
            lens_front=StandardSurface(),
            lens_back=StandardSurface(),
            retina=StandardSurface(),
        )

    def test_init_geometry_non_stop_pupil_raises_valueerror(self):
        with pytest.raises(ValueError, match="The pupil surface must be a stop"):
            EyeGeometry(
                cornea_front=StandardSurface(),
                cornea_back=StandardSurface(),
                pupil=StandardSurface(is_stop=False),
                lens_front=StandardSurface(),
                lens_back=StandardSurface(),
                retina=StandardSurface(),
            )

    @pytest.mark.parametrize("asphericity", [-1, -1.5])
    def test_init_geometry_non_ellipsoid_retina_raises_valueerror(self, asphericity):
        with pytest.raises(ValueError, match="Only an elliptical retina is allowed"):
            EyeGeometry(
                cornea_front=StandardSurface(),
                cornea_back=StandardSurface(),
                pupil=Stop(),
                lens_front=StandardSurface(),
                lens_back=StandardSurface(),
                retina=StandardSurface(radius=12, asphericity=asphericity),
            )

    def test_axial_length(self, example_geometry_parameters, example_geometry):
        assert example_geometry.axial_length == example_geometry_parameters["axial_length"]

    def test_cornea_thickness(self, example_geometry_parameters, example_geometry):
        assert example_geometry.cornea_thickness == example_geometry_parameters["cornea_thickness"]

    def test_anterior_chamber_depth(self, example_geometry_parameters, example_geometry):
        assert example_geometry.anterior_chamber_depth == example_geometry_parameters["anterior_chamber_depth"]

    def test_pupil_lens_distance(self, example_geometry_parameters, example_geometry):
        assert example_geometry.pupil_lens_distance == example_geometry_parameters["pupil_lens_distance"]

    def test_lens_thickness(self, example_geometry_parameters, example_geometry):
        assert example_geometry.lens_thickness == example_geometry_parameters["lens_thickness"]

    def test_vitreous_thickness(self, example_geometry_parameters, example_geometry):
        vitreous_thickness = example_geometry_parameters["axial_length"] - (
            example_geometry_parameters["cornea_thickness"]
            + example_geometry_parameters["anterior_chamber_depth"]
            + example_geometry_parameters["pupil_lens_distance"]
            + example_geometry_parameters["lens_thickness"]
        )
        assert example_geometry.vitreous_thickness == vitreous_thickness
