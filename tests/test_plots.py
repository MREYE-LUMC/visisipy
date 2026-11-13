"""Tests for the plotting functionality."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from visisipy.models.catalog.navarro import NavarroGeometry
from visisipy.models.geometry import BiconicSurface, EvenAsphereSurface, EyeGeometry, StandardSurface, Stop
from visisipy.plots import _backend_translation, _plot_ellipse, _plot_hyperbola, _plot_parabola, plot_eye

mpl.use("Agg")  # use non-interactive backend for testing


def _plot_eye_path(geometry: EyeGeometry) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Plot an eye and return the vertices of the plotted path and the axis limits."""
    fig, ax = plt.subplots()
    plot_eye(ax, geometry, backend="opticstudio")

    vertices = ax.patches[0].get_path().vertices
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)

    return vertices, xlim, ylim


class TestConicSections:
    """Test individual conic section plotting functions."""

    def test_plot_ellipse_normal_cutoff(self):
        """Test ellipse plotting with a normal cutoff."""
        path = _plot_ellipse(position=0, radius=10, conic=0.5, cutoff=5)
        assert path is not None
        assert len(path.vertices) > 0

    def test_plot_ellipse_out_of_range_cutoff(self):
        """Test ellipse plotting with cutoff beyond surface extent."""
        with pytest.raises(ValueError, match="Cutoff is located outside the ellipse"):
            _plot_ellipse(position=0, radius=10, conic=0.5, cutoff=100)

    def test_plot_ellipse_negative_radius(self):
        """Test ellipse plotting with negative radius (reversed orientation)."""
        path = _plot_ellipse(position=0, radius=-10, conic=0.5, cutoff=-5)
        assert path is not None
        assert len(path.vertices) > 0

    def test_plot_ellipse_with_endpoint(self):
        """Test ellipse plotting with endpoint return."""
        path, endpoint_y = _plot_ellipse(position=0, radius=10, conic=0.5, cutoff=5, return_endpoint=True)
        assert path is not None
        assert isinstance(endpoint_y, (float, np.floating))

    def test_plot_parabola_normal_cutoff(self):
        """Test parabola plotting with a normal cutoff."""
        path = _plot_parabola(position=0, radius=10, cutoff=5)
        assert path is not None
        assert len(path.vertices) > 0

    def test_plot_parabola_out_of_range_cutoff(self):
        """Test parabola plotting with cutoff beyond domain."""
        with pytest.raises(ValueError, match="Cutoff is outside the domain of the parabola"):
            _plot_parabola(position=0, radius=10, cutoff=-10)

    def test_plot_parabola_with_endpoint(self):
        """Test parabola plotting with endpoint return."""
        path, endpoint_y = _plot_parabola(position=0, radius=10, cutoff=5, return_endpoint=True)
        assert path is not None
        assert isinstance(endpoint_y, (float, np.floating))

    def test_plot_hyperbola_normal_cutoff(self):
        """Test hyperbola plotting with a normal cutoff."""
        path = _plot_hyperbola(position=0, radius=10, conic=-2, cutoff=5)
        assert path is not None
        assert len(path.vertices) > 0

    def test_plot_hyperbola_out_of_range_cutoff(self):
        """Test hyperbola plotting with cutoff beyond domain."""
        with pytest.raises(ValueError, match="The cutoff coordinate is located outside the domain of the hyperbola"):
            _plot_hyperbola(position=0, radius=10, conic=-2, cutoff=-50)

    def test_plot_hyperbola_with_endpoint(self):
        """Test hyperbola plotting with endpoint return."""
        path, endpoint_y = _plot_hyperbola(position=0, radius=10, conic=-2, cutoff=5, return_endpoint=True)
        assert path is not None
        assert isinstance(endpoint_y, (float, np.floating))


class TestPlotEye:
    """Test the plot_eye function with various eye configurations."""

    def test_default_navarro(self):
        """Test plotting a default Navarro eye."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    def test_lens_edge_thickness(self):
        """Test plotting an eye with lens edge thickness."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry, lens_edge_thickness=0.5)
        assert result_ax is ax
        plt.close(fig)

    def test_retina_cutoff(self):
        """Test plotting an eye with custom retina cutoff."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry, retina_cutoff_position=2.0)
        assert result_ax is ax
        plt.close(fig)

    def test_positive_retina_radius(self):
        """Test plotting an eye with incorrectly oriented retina (positive radius)."""
        # This should now work without crashing
        geometry = NavarroGeometry()
        # Override the retina validation to test robustness
        geometry.retina = StandardSurface(radius=12.0, asphericity=0)  # Positive radius

        fig, ax = plt.subplots()

        with pytest.warns(UserWarning, match="Convex retina detected. Drawing to maximum thickness of 5 mm"):
            result_ax = plot_eye(ax, geometry)

        assert result_ax is ax
        plt.close(fig)

    def test_strongly_curved_retina(self):
        """Test plotting an eye with a very strongly curved retina that might not intersect lens."""
        geometry = NavarroGeometry()
        # Create a very strongly curved retina
        geometry.retina = StandardSurface(radius=-5.0, asphericity=0)

        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    def test_non_intersecting_lens_surfaces(self):
        """Test plotting an eye where lens front and back don't intersect."""
        geometry = EyeGeometry(
            cornea_front=StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55),
            cornea_back=StandardSurface(radius=6.50, asphericity=0, thickness=3.05),
            pupil=Stop(semi_diameter=1.348),
            lens_front=StandardSurface(radius=2.0, asphericity=0, thickness=4.0),  # Very flat
            lens_back=StandardSurface(radius=-2.0, asphericity=0, thickness=16.3203),  # Very flat
            retina=StandardSurface(radius=-12.0, asphericity=0),
        )

        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    def test_parabolic_surface(self):
        """Test plotting an eye with a parabolic surface (conic = -1)."""
        geometry = NavarroGeometry()
        geometry.lens_back.asphericity = -1  # Make it parabolic

        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    def test_hyperbolic_surface(self):
        """Test plotting an eye with a hyperbolic surface (conic < -1)."""
        geometry = NavarroGeometry()
        geometry.lens_front.asphericity = -2  # Make it hyperbolic

        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    def test_invalid_lens_edge_thickness(self):
        """Test that negative lens edge thickness raises ValueError."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="lens_edge_thickness should be a positive number"):
            plot_eye(ax, geometry, lens_edge_thickness=-1.0)

        plt.close(fig)

    def test_invalid_retina_cutoff(self):
        """Test that retina cutoff behind retina raises ValueError."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="retina_cutoff_position is located behind the retina"):
            plot_eye(ax, geometry, retina_cutoff_position=100.0)

        plt.close(fig)

    def test_extreme_asphericities(self):
        """Test plotting with extreme asphericity values."""
        geometry = EyeGeometry(
            cornea_front=StandardSurface(radius=7.72, asphericity=-0.9, thickness=0.55),
            cornea_back=StandardSurface(radius=6.50, asphericity=0.9, thickness=3.05),
            pupil=Stop(semi_diameter=1.348),
            lens_front=StandardSurface(radius=10.2, asphericity=-0.5, thickness=4.0),
            lens_back=StandardSurface(radius=-6.0, asphericity=-1.5, thickness=16.3203),
            retina=StandardSurface(radius=-12.0, asphericity=0.5),
        )

        fig, ax = plt.subplots()
        result_ax = plot_eye(ax, geometry)
        assert result_ax is ax
        plt.close(fig)

    @pytest.mark.parametrize(
        "lens_back_radius,retina_radius",
        [
            pytest.param(-6.0, -12.0, id="normal-retina"),
            pytest.param(-6.0, -5.0, id="strongly-curved-retina"),
            pytest.param(6.0, -12.0, id="concave-lens-back"),
        ],
    )
    @pytest.mark.parametrize(
        "retina_type",
        [
            pytest.param("biconic", id="biconic-retina"),
            pytest.param("even-asphere", id="even-asphere-retina"),
        ],
    )
    def test_non_ellipsoid_retina_plotted_as_base_conic(self, lens_back_radius, retina_radius, retina_type):
        """Test that retinas that are not ellipsoids are plotted as their base conic in the y-z plane."""
        asphericity = 0.2

        match retina_type:
            case "biconic":
                retina = BiconicSurface(
                    radius=retina_radius, asphericity=asphericity, radius_x=retina_radius + 1, asphericity_x=asphericity
                )
            case "even-asphere":
                retina = EvenAsphereSurface(
                    radius=retina_radius, asphericity=asphericity, asphere_coefficients=[0.1, 0.01, 0.001]
                )

        with pytest.raises(NotImplementedError):
            _ = retina.ellipsoid_radii

        lens_back = StandardSurface(radius=lens_back_radius, asphericity=0, thickness=16.3203)
        base_conic_retina = StandardSurface(radius=retina_radius, asphericity=asphericity)

        vertices, xlim, ylim = _plot_eye_path(NavarroGeometry(lens_back=lens_back, retina=retina))
        expected_vertices, expected_xlim, expected_ylim = _plot_eye_path(
            NavarroGeometry(lens_back=lens_back, retina=base_conic_retina)
        )

        np.testing.assert_allclose(vertices, expected_vertices)
        assert xlim == pytest.approx(expected_xlim)
        assert ylim == pytest.approx(expected_ylim)

    def test_all_surfaces_meet_at_apexes(self):
        """Test that surfaces are correctly cut off at adjacent apex positions."""
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()
        _ = plot_eye(ax, geometry)

        # Get the compound path
        patches = ax.patches
        assert len(patches) > 0

        # The patch should have been successfully created
        compound_path = patches[0].get_path()
        assert compound_path is not None

        plt.close(fig)


class TestBackendTranslation:
    def test_opticstudio_no_translation(self):
        geometry = NavarroGeometry()

        translation = _backend_translation(geometry, "opticstudio")

        assert translation == 0.0

    def test_optiland_translation(self):
        geometry = NavarroGeometry()

        translation = _backend_translation(geometry, "optiland")

        assert translation == geometry.cornea_front.thickness + geometry.cornea_back.thickness

    def test_unknown_backend_translation(self):
        geometry = NavarroGeometry()
        fig, ax = plt.subplots()

        with pytest.warns(UserWarning, match="Unknown backend unknown_backend. No translation applied."):
            plot_eye(ax, geometry, backend="unknown_backend")

        plt.close(fig)
