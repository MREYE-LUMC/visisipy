from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from visisipy.models import EyeGeometry, NavarroGeometry, create_geometry
from visisipy.models.geometry import (
    BaseZernikeStandardSurface,
    BiconicSurface,
    GeometryParameters,
    StandardSurface,
    Stop,
    ZernikeStandardPhaseSurface,
    ZernikeStandardSagSurface,
)


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


@pytest.mark.parametrize("base_geometry", [NavarroGeometry])
class TestCreateGeometry:
    def test_create_geometry(self, base_geometry, example_geometry_parameters, example_geometry):
        class SentinelFloat(float):
            """Custom float class to mark floats as set by a unit test."""

        geometry = create_geometry(
            base=base_geometry,
            **{k: SentinelFloat(v) for k, v in example_geometry_parameters.items()},
            estimate_cornea_back=False,
        )

        assert geometry.__dict__ == example_geometry.__dict__

        assert isinstance(geometry.cornea_front.thickness, SentinelFloat)
        assert isinstance(geometry.cornea_front.radius, SentinelFloat)
        assert isinstance(geometry.cornea_front.asphericity, SentinelFloat)

        assert isinstance(geometry.cornea_back.radius, SentinelFloat)
        assert isinstance(geometry.cornea_back.asphericity, SentinelFloat)
        assert isinstance(geometry.cornea_back.thickness, SentinelFloat)

        assert isinstance(geometry.pupil.semi_diameter, SentinelFloat)
        assert isinstance(geometry.pupil.thickness, SentinelFloat)

        assert isinstance(geometry.lens_front.thickness, SentinelFloat)
        assert isinstance(geometry.lens_front.radius, SentinelFloat)
        assert isinstance(geometry.lens_front.asphericity, SentinelFloat)

        # Note: the lens back thickness is a calculated value, so it is not a SentinelFloat
        assert isinstance(geometry.lens_back.radius, SentinelFloat)
        assert isinstance(geometry.lens_back.asphericity, SentinelFloat)

        assert isinstance(geometry.retina.radius, SentinelFloat)
        assert isinstance(geometry.retina.asphericity, SentinelFloat)

    def test_create_geometry_estimate_cornea_back(self, base_geometry, example_geometry_parameters):
        with pytest.warns(
            UserWarning,
            match="The cornea back radius was provided, but it will be ignored",
        ):
            geometry = create_geometry(
                base=base_geometry,
                **example_geometry_parameters,
                estimate_cornea_back=True,
            )

        assert geometry.cornea_back.radius == 0.81 * geometry.cornea_front.radius

    def test_base_no_class_raises_typeerror(self, base_geometry):
        with pytest.raises(TypeError, match="The base geometry must be a class."):
            create_geometry(base=base_geometry())

    def test_base_no_eyegeometry_raises_typeerror(self, base_geometry):
        class InvalidGeometry: ...

        with pytest.raises(TypeError, match="The base geometry must be a subclass of EyeGeometry."):
            create_geometry(base=InvalidGeometry)

    def test_estimate_cornea_back_and_radius_warns(self, base_geometry, example_geometry_parameters):
        with pytest.warns(
            UserWarning,
            match="The cornea back radius was provided, but it will be ignored because estimate_cornea_back is True",
        ):
            create_geometry(
                base=base_geometry,
                estimate_cornea_back=True,
                **example_geometry_parameters,
            )

    @pytest.mark.parametrize(
        "parameters_a",
        [
            {"retina_radius": 12},
            {"retina_asphericity": 0},
            {"retina_radius": 12, "retina_asphericity": 0},
        ],
    )
    @pytest.mark.parametrize(
        "parameters_b",
        [
            {"retina_ellipsoid_z_radius": 12},
            {"retina_ellipsoid_y_radius": 12},
            {"retina_ellipsoid_z_radius": 12, "retina_ellipsoid_y_radius": 12},
        ],
    )
    def test_supplying_retina_parameters_and_ellipsoid_radii_raises_valueerror(
        self, base_geometry, parameters_a, parameters_b
    ):
        parameters = parameters_a | parameters_b

        with pytest.raises(ValueError, match="Cannot specify both retina radius/asphericity and ellipsoid radii"):
            create_geometry(base=base_geometry, **parameters)

    @pytest.mark.parametrize(
        "parameters",
        [
            {"retina_ellipsoid_z_radius": 12},
            {"retina_ellipsoid_y_radius": 12},
        ],
    )
    def test_supplying_single_ellipsoid_radius_raises_valueerror(self, base_geometry, parameters):
        with pytest.raises(
            ValueError,
            match="If the retina ellipsoid radii are specified, both the z and y radius must be provided",
        ):
            create_geometry(base=base_geometry, **parameters)

    def test_incomplete_thickness_raises_valueerror(self, base_geometry, monkeypatch):
        # Mock the __init__ method to avoid setting the thickness of the cornea front
        def mock_init(self, *args, **kwargs):
            self.cornea_front = StandardSurface(thickness=None)
            self.cornea_back = StandardSurface()
            self.pupil = Stop()
            self.lens_front = StandardSurface()
            self.lens_back = StandardSurface()
            self.retina = StandardSurface()

        monkeypatch.setattr(base_geometry, "__init__", mock_init)

        with pytest.raises(
            ValueError,
            match="Cannot calculate vitreous thickness from the supplied parameters.",
        ):
            create_geometry(base_geometry, axial_length=20)

    @pytest.mark.parametrize(
        "geometry_parameters",
        [
            {"axial_length": 10, "cornea_thickness": 1, "anterior_chamber_depth": 4, "lens_thickness": 5},
            {"axial_length": 10, "cornea_thickness": 2, "anterior_chamber_depth": 4, "lens_thickness": 5},
        ],
    )
    def test_invalid_vitreous_thickness_raises_valueerror(self, base_geometry, geometry_parameters):
        with pytest.raises(
            ValueError,
            match="The sum of the cornea thickness, anterior chamber depth, pupil-lens distance and lens thickness is "
            "greater than or equal to the axial length.",
        ):
            create_geometry(
                base=base_geometry,
                axial_length=10,
                cornea_thickness=1,
                anterior_chamber_depth=4,
                lens_thickness=5,
            )

    def test_set_retina_ellipsoid_radii(self, base_geometry):
        geometry = create_geometry(base=base_geometry, retina_ellipsoid_z_radius=12.34, retina_ellipsoid_y_radius=10)

        assert pytest.approx(geometry.retina.ellipsoid_radii.z) == 12.34
        assert pytest.approx(geometry.retina.ellipsoid_radii.y) == 10
        assert pytest.approx(geometry.retina.ellipsoid_radii.x) == 10
