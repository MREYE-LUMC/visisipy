from __future__ import annotations

import numpy as np
import pytest

from visisipy.models import EyeGeometry, NavarroGeometry, create_geometry
from visisipy.models.geometry import StandardSurface, Stop


@pytest.fixture
def example_geometry_parameters():
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
        pupil=Stop(semi_diameter=example_geometry_parameters["pupil_radius"]),
        lens_front=StandardSurface(
            radius=example_geometry_parameters["lens_front_radius"],
            asphericity=example_geometry_parameters["lens_front_asphericity"],
            thickness=example_geometry_parameters["lens_thickness"],
        ),
        lens_back=StandardSurface(
            radius=example_geometry_parameters["lens_back_radius"],
            asphericity=example_geometry_parameters["lens_back_asphericity"],
            thickness=(
                example_geometry_parameters["axial_length"]
                - example_geometry_parameters["cornea_thickness"]
                - example_geometry_parameters["anterior_chamber_depth"]
                - example_geometry_parameters["lens_thickness"]
            ),
        ),
        retina=StandardSurface(
            radius=example_geometry_parameters["retina_radius"],
            asphericity=example_geometry_parameters["retina_asphericity"],
        ),
    )


class TestStandardSurface:
    @pytest.mark.parametrize(
        "radius,asphericity,axial_half_axis,radial_half_axis",
        [
            (12, 0, 12, 12),
            (12, 1, 6, 12 / np.sqrt(2)),
            (12, -0.5, 24, 12 / np.sqrt(0.5)),
        ],
    )
    def test_half_axes(self, radius, asphericity, axial_half_axis, radial_half_axis):
        surface = StandardSurface(radius=radius, asphericity=asphericity)

        assert surface.half_axes == pytest.approx((axial_half_axis, radial_half_axis))
        assert surface.half_axes.axial == pytest.approx(axial_half_axis)
        assert surface.half_axes.radial == pytest.approx(radial_half_axis)

    @pytest.mark.parametrize("asphericity", [-1, -1.5])
    def test_half_axes_raises_notimplementederror(self, asphericity):
        surface = StandardSurface(radius=12, asphericity=asphericity)

        with pytest.raises(NotImplementedError):
            _ = surface.half_axes


class TestStop:
    def test_init_stop(self):
        stop = Stop()

        assert stop.is_stop
        assert stop.semi_diameter > 0


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

    def test_lens_thickness(self, example_geometry_parameters, example_geometry):
        assert example_geometry.lens_thickness == example_geometry_parameters["lens_thickness"]

    def test_vitreous_thickness(self, example_geometry_parameters, example_geometry):
        vitreous_thickness = example_geometry_parameters["axial_length"] - (
            example_geometry_parameters["cornea_thickness"]
            + example_geometry_parameters["anterior_chamber_depth"]
            + example_geometry_parameters["lens_thickness"]
        )
        assert example_geometry.vitreous_thickness == vitreous_thickness


@pytest.mark.parametrize("base_geometry", [NavarroGeometry])
class TestCreateGeometry:
    def test_create_geometry(self, base_geometry, example_geometry_parameters, example_geometry):
        geometry = create_geometry(
            base=base_geometry,
            **example_geometry_parameters,
            estimate_cornea_back=False,
        )

        assert geometry.__dict__ == example_geometry.__dict__

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
            match="The sum of the cornea thickness, anterior chamber depth and lens thickness is greater than "
            "or equal to the axial length.",
        ):
            create_geometry(
                base=base_geometry,
                axial_length=10,
                cornea_thickness=1,
                anterior_chamber_depth=4,
                lens_thickness=5,
            )
