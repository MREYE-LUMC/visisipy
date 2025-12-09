from __future__ import annotations

import pytest

from visisipy.models.catalog.navarro import NavarroGeometry
from visisipy.models.geometry import StandardSurface, Stop, ZernikeStandardSagSurface


class TestNavarroGeometry:
    def test_create_navarro_geometry(self):
        geometry = NavarroGeometry()

        assert isinstance(geometry.cornea_front, StandardSurface)
        assert geometry.cornea_front.radius == 7.72
        assert geometry.cornea_front.asphericity == -0.26
        assert geometry.cornea_front.thickness == 0.55
        assert isinstance(geometry.cornea_back, StandardSurface)
        assert geometry.cornea_back.radius == 6.50
        assert geometry.cornea_back.asphericity == 0
        assert geometry.cornea_back.thickness == 3.05
        assert isinstance(geometry.pupil, Stop)
        assert geometry.pupil.semi_diameter == 1.348
        assert geometry.pupil.thickness == 0
        assert isinstance(geometry.lens_front, StandardSurface)
        assert geometry.lens_front.radius == 10.2
        assert geometry.lens_front.asphericity == -3.1316
        assert geometry.lens_front.thickness == 4.0
        assert isinstance(geometry.lens_back, StandardSurface)
        assert geometry.lens_back.radius == -6.0
        assert geometry.lens_back.asphericity == -1
        assert geometry.lens_back.thickness == 16.3203
        assert isinstance(geometry.retina, StandardSurface)
        assert geometry.retina.radius == -12.0
        assert geometry.retina.asphericity == 0
        assert geometry.retina.thickness == 0

        assert geometry.axial_length == pytest.approx(23.9203)

    def test_create_navarro_geometry_with_custom_surface(self):
        geometry = NavarroGeometry(
            lens_back=ZernikeStandardSagSurface(radius=-5.8, asphericity=-0.9, thickness=18.3203)
        )

        assert isinstance(geometry.lens_back, ZernikeStandardSagSurface)
        assert geometry.lens_back.radius == -5.8
        assert geometry.lens_back.asphericity == -0.9
        assert geometry.lens_back.thickness == 18.3203

        assert geometry.axial_length == pytest.approx(25.9203)
