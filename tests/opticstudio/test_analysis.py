import pytest


pytestmark = [pytest.mark.needs_opticstudio]


@pytest.fixture
def opticstudio_analysis(opticstudio_backend):
    return opticstudio_backend.analysis


class TestRayTraceAnalysis:
    def test_raytrace(self, opticstudio_analysis):
        coordinates = [(0, 0), (1, 1)]

        assert opticstudio_analysis.raytrace(coordinates)

    def test_raytrace_dataframe_structure(self, opticstudio_analysis):
        coordinates = [(0, 0), (1, 1)]

        result, _ = opticstudio_analysis.raytrace(coordinates)
        expected_columns = {
            "index",
            "x",
            "y",
            "z",
            "field",
            "wavelength",
            "surface",
            "comment",
        }

        assert set(result.columns) == expected_columns
        assert set(result.field.unique()) == set(coordinates)


class TestZernikeStandardCoefficientsAnalysis:
    @pytest.mark.parametrize(
        "field_coordinate,wavelength", [(None, None), ((0, 0), 0.543), ((1, 1), 0.632)]
    )
    def test_zernike_standard_coefficients(
        self, opticstudio_analysis, field_coordinate, wavelength
    ):
        assert opticstudio_analysis.zernike_standard_coefficients(
            field_coordinate=field_coordinate, wavelength=wavelength
        )


class MockPupil:
    def __init__(self):
        self.semi_diameter = 1.0
        self.changed_semi_diameter = False

    @property
    def semi_diameter(self):
        return self._semi_diameter

    @semi_diameter.setter
    def semi_diameter(self, value):
        self.changed_semi_diameter = True
        self._semi_diameter = value


class MockOpticstudioModel:
    def __init__(self):
        self._pupil = MockPupil()

    @property
    def pupil(self):
        return self._pupil


class TestRefractionAnalysis:
    @pytest.mark.parametrize(
        "use_higher_order_aberrations,wavelength,pupil_diameter",
        [
            (False, None, None),
            (False, 0.543, 0.5),
            (True, 0.543, 1),
        ],
    )
    def test_refraction(
        self,
        opticstudio_analysis,
        use_higher_order_aberrations,
        wavelength,
        pupil_diameter,
        monkeypatch,
    ):
        monkeypatch.setattr(
            opticstudio_analysis._backend, "model", MockOpticstudioModel()
        )

        assert opticstudio_analysis.refraction(
            use_higher_order_aberrations=use_higher_order_aberrations,
            wavelength=wavelength,
            pupil_diameter=pupil_diameter,
        )

    @pytest.mark.parametrize(
        "pupil_diameter,changed_pupil_diameter",
        [
            (None, False),
            (0.5, True),
        ],
    )
    def test_refraction_change_pupil(
        self, opticstudio_analysis, pupil_diameter, changed_pupil_diameter, monkeypatch
    ):
        monkeypatch.setattr(
            opticstudio_analysis._backend, "model", MockOpticstudioModel()
        )

        assert not opticstudio_analysis._backend.model.pupil.changed_semi_diameter

        opticstudio_analysis.refraction(pupil_diameter=pupil_diameter)

        assert (
            opticstudio_analysis._backend.model.pupil.changed_semi_diameter
            == changed_pupil_diameter
        )
        assert opticstudio_analysis._backend.model.pupil.semi_diameter == 1.0
