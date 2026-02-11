from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

import visisipy

if TYPE_CHECKING:
    from visisipy.backend import BaseBackend
    from visisipy.models.base import EyeModel


class BaseAnalysisTest(ABC):
    @abstractmethod
    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame: ...

    """Run the analysis to generate test data.

    This method must return a dataframe with the results of the analysis.
    Series are intentionally not supported, to make loading the results easier and more consistent across tests.
    """


class FFTPSFTest(BaseAnalysisTest):
    def __init__(
        self,
        coordinate: tuple[float, float],
        sampling: int,
        wavelength: float,
    ) -> None:
        self.coordinate = coordinate
        self.sampling = sampling
        self.wavelength = wavelength

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        return visisipy.analysis.fft_psf(
            model=model,
            field_coordinate=self.coordinate,
            field_type="angle",
            sampling=self.sampling,
            wavelength=self.wavelength,
            backend=backend,
        )


class HuygensPSFTest(BaseAnalysisTest):
    def __init__(
        self,
        coordinate: tuple[float, float],
        pupil_sampling: int,
        image_sampling: int,
        wavelength: float,
    ) -> None:
        self.coordinate = coordinate
        self.pupil_sampling = pupil_sampling
        self.image_sampling = image_sampling
        self.wavelength = wavelength

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        return visisipy.analysis.huygens_psf(
            model=model,
            field_coordinate=self.coordinate,
            field_type="angle",
            pupil_sampling=self.pupil_sampling,
            image_sampling=self.image_sampling,
            wavelength=self.wavelength,
            backend=backend,
        )


class RayTraceTest(BaseAnalysisTest):
    def __init__(
        self,
        coordinates: list[tuple[float, float]],
        pupil: tuple[float, float],
        sampling: int,
        wavelength: float,
    ) -> None:
        self.coordinates = coordinates
        self.pupil = pupil
        self.sampling = sampling
        self.wavelength = wavelength

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        return visisipy.analysis.raytrace(
            model=model,
            coordinates=self.coordinates,
            field_type="angle",
            wavelengths=[self.wavelength],
            pupil=self.pupil,
            backend=backend,
        )


class RefractionTest(BaseAnalysisTest):
    def __init__(
        self,
        *,
        coordinates: list[tuple[float, float]],
        wavelength: float,
        sampling: int,
    ) -> None:
        self.coordinates = coordinates
        self.sampling = sampling
        self.wavelength = wavelength

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        result: list[tuple[float, float, float, float, float]] = []

        for coord in self.coordinates:
            # Force building a new model for each coordinate to ensure that the backend is properly cleared and reset between runs.
            backend.clear_model()

            refraction = visisipy.analysis.refraction(
                model=model,
                field_coordinate=coord,
                sampling=self.sampling,
                wavelength=self.wavelength,
                backend=backend,
            )

            result.append((coord[0], coord[1], refraction.M, refraction.J0, refraction.J45))

        return pd.DataFrame(result, columns=["field_x", "field_y", "M", "J0", "J45"])


class OPDMapTest(BaseAnalysisTest):
    def __init__(
        self,
        *,
        coordinate: tuple[float, float],
        sampling: int,
        wavelength: float,
        remove_tilt: bool,
    ) -> None:
        self.coordinate = coordinate
        self.sampling = sampling
        self.wavelength = wavelength
        self.remove_tilt = remove_tilt

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        return visisipy.analysis.opd_map(
            model=model,
            field_coordinate=self.coordinate,
            wavelength=self.wavelength,
            field_type="angle",
            sampling=self.sampling,
            remove_tilt=self.remove_tilt,
            use_exit_pupil_shape=False,
            backend=backend,
        )


class ZernikeStandardCoefficientsTest(BaseAnalysisTest):
    def __init__(
        self,
        *,
        coordinate: tuple[float, float],
        sampling: int,
        wavelength: float,
        maximum_term: int,
    ) -> None:
        self.coordinate = coordinate
        self.sampling = sampling
        self.wavelength = wavelength
        self.maximum_term = maximum_term

    def run(self, model: EyeModel, backend: type[BaseBackend]) -> pd.DataFrame:
        result = visisipy.analysis.zernike_standard_coefficients(
            model=model,
            field_coordinate=self.coordinate,
            wavelength=self.wavelength,
            field_type="angle",
            sampling=self.sampling,
            maximum_term=self.maximum_term,
            unit="microns",
            backend=backend,
        )

        return pd.DataFrame(result.items(), columns=["coefficient", "value"])
