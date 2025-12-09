from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise

import pytest

from visisipy.models.catalog.navarro import NavarroGeometry
from visisipy.models.factory import _check_sign, create_geometry
from visisipy.models.geometry import StandardSurface, Stop


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
            {"retina_radius": -12},
            {"retina_asphericity": 0},
            {"retina_radius": -12, "retina_asphericity": 0},
        ],
    )
    @pytest.mark.parametrize(
        "parameters_b",
        [
            {"retina_ellipsoid_z_radius": -12},
            {"retina_ellipsoid_y_radius": 12},
            {"retina_ellipsoid_z_radius": -12, "retina_ellipsoid_y_radius": 12},
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
            {"retina_ellipsoid_z_radius": -12},
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
        geometry = create_geometry(base=base_geometry, retina_ellipsoid_z_radius=-12.34, retina_ellipsoid_y_radius=10)

        assert pytest.approx(geometry.retina.ellipsoid_radii.z) == -12.34
        assert pytest.approx(geometry.retina.ellipsoid_radii.y) == 10
        assert pytest.approx(geometry.retina.ellipsoid_radii.x) == 10


class TestCheckSign:
    @pytest.mark.parametrize(
        "sign,value,expectation",
        [
            ("+", 5, does_not_raise()),
            ("+", 0, does_not_raise()),
            (
                "+",
                -5,
                pytest.warns(
                    UserWarning,
                    match=re.escape("Expected a positive value for test_param, got -5. Check if the sign is correct."),
                ),
            ),
            ("-", -5, does_not_raise()),
            ("-", 0, does_not_raise()),
            (
                "-",
                5,
                pytest.warns(
                    UserWarning,
                    match=re.escape("Expected a negative value for test_param, got 5. Check if the sign is correct."),
                ),
            ),
            (
                "invalid",
                0,
                pytest.raises(
                    ValueError, match=re.escape("Invalid sign 'invalid' specified for test_param. Must be '+' or '-'.")
                ),
            ),
        ],
    )
    @pytest.mark.filterwarnings("error")  # Fail on unexpected warnings
    def test_check_sign(self, sign, value, expectation):
        with expectation:
            _check_sign(value, "test_param", sign)

    @pytest.mark.parametrize(
        "parameter,value,expectation",
        [
            ("cornea_front_radius", 1, does_not_raise()),
            (
                "cornea_front_radius",
                -1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a positive value for cornea_front_radius, got -1. Check if the sign is correct."
                    ),
                ),
            ),
            ("cornea_back_radius", 1, does_not_raise()),
            (
                "cornea_back_radius",
                -1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a positive value for cornea_back_radius, got -1. Check if the sign is correct."
                    ),
                ),
            ),
            ("lens_front_radius", 1, does_not_raise()),
            (
                "lens_front_radius",
                -1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a positive value for lens_front_radius, got -1. Check if the sign is correct."
                    ),
                ),
            ),
            ("lens_back_radius", -1, does_not_raise()),
            (
                "lens_back_radius",
                1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a negative value for lens_back_radius, got 1. Check if the sign is correct."
                    ),
                ),
            ),
            ("retina_radius", -1, does_not_raise()),
            (
                "retina_radius",
                1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a negative value for retina_radius, got 1. Check if the sign is correct."
                    ),
                ),
            ),
            ("retina_ellipsoid_z_radius", -1, does_not_raise()),
            (
                "retina_ellipsoid_z_radius",
                1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a negative value for retina_ellipsoid_z_radius, got 1. Check if the sign is correct."
                    ),
                ),
            ),
            ("retina_ellipsoid_y_radius", 1, does_not_raise()),
            (
                "retina_ellipsoid_y_radius",
                -1,
                pytest.warns(
                    UserWarning,
                    match=re.escape(
                        "Expected a positive value for retina_ellipsoid_y_radius, got -1. Check if the sign is correct."
                    ),
                ),
            ),
        ],
    )
    @pytest.mark.filterwarnings("error")  # Fail on unexpected warnings
    def test_create_geometry_checks_signs(self, parameter, value, expectation):
        if parameter == "retina_ellipsoid_z_radius":
            params = {"retina_ellipsoid_y_radius": 1, parameter: value}
        elif parameter == "retina_ellipsoid_y_radius":
            params = {"retina_ellipsoid_z_radius": -1, parameter: value}
        else:
            params = {parameter: value}

        with expectation:
            create_geometry(**params)
