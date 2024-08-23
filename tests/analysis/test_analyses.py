import visisipy
import visisipy.backend


class MockAnalysis:
    def cardinal_points(
        self,
        surface_1,
        surface_2,
    ):
        return None, None

    def raytrace(
        self,
        coordinates,
        wavelengths,
        field_type,
        pupil,
    ):
        return None, None

    def zernike_standard_coefficients(self, field_coordinate, wavelength, field_type, sampling, maximum_term):
        return None, None

    def refraction(
        self,
        use_higher_order_aberrations,
        field_coordinate,
        wavelength,
        pupil_diameter,
        field_type,
    ):
        return None, None


class MockBackend:
    analysis = MockAnalysis()


def test_cardinal_points_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.cardinal_points() is None
    assert visisipy.analysis.cardinal_points(return_raw_result=True) == (None, None)


def test_raytracing_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.raytrace(coordinates=[(0, 0)]) is None
    assert visisipy.analysis.raytrace(coordinates=[(0, 0)], return_raw_result=True) == (
        None,
        None,
    )


def test_zernike_standard_coefficients_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.zernike_standard_coefficients() is None
    assert visisipy.analysis.zernike_standard_coefficients(return_raw_result=True) == (
        None,
        None,
    )


def test_refraction_analysis(monkeypatch):
    monkeypatch.setattr(visisipy.backend, "_BACKEND", MockBackend)

    assert visisipy.analysis.refraction() is None
    assert visisipy.analysis.refraction(return_raw_result=True) == (None, None)
