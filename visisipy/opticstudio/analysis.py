from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import zospy as zp
from pandas import DataFrame

from visisipy.analysis.cardinal_points import CardinalPoints, CardinalPointsResult
from visisipy.backend import BaseAnalysis
from visisipy.refraction import FourierPowerVectorRefraction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zospy.api import _ZOSAPI
    from zospy.zpcore import OpticStudioSystem

    from visisipy.opticstudio.backend import OpticStudioBackend

__all__ = ("OpticStudioAnalysis",)


def _iter_fields(oss: OpticStudioSystem) -> tuple[int, _ZOSAPI.SystemData.IField]:
    for i in range(oss.SystemData.Fields.NumberOfFields):
        field = oss.SystemData.Fields.GetField(i + 1)
        yield field.FieldNumber, field


def _iter_wavelengths(oss: OpticStudioSystem) -> tuple[int, float]:
    for i in range(oss.SystemData.Wavelengths.NumberOfWavelengths):
        yield i + 1, oss.SystemData.Wavelengths.GetWavelength(i + 1)


def _build_cardinal_points_result(cardinal_points_result: zp.analyses.base.AttrDict) -> CardinalPointsResult:
    return CardinalPointsResult(
        focal_lengths=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Focal Length"],
            object=cardinal_points_result.Data["Object Space"]["Focal Length"],
        ),
        focal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Focal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Focal Planes"],
        ),
        principal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Principal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Principal Planes"],
        ),
        anti_principal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Anti-Principal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Anti-Principal Planes"],
        ),
        anti_nodal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Anti-Nodal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Anti-Nodal Planes"],
        ),
        nodal_points=CardinalPoints(
            image=cardinal_points_result.Data["Image Space"]["Nodal Planes"],
            object=cardinal_points_result.Data["Object Space"]["Nodal Planes"],
        ),
    )


def _build_raytrace_result(raytrace_results: list[DataFrame]) -> DataFrame:
    columns = {
        "Field": "field",
        "Wavelength": "wavelength",
        "Surf": "surface",
        "Comment": "comment",
        "X-coordinate": "x",
        "Y-coordinate": "y",
        "Z-coordinate": "z",
    }

    return pd.concat(raytrace_results)[columns.keys()].rename(columns=columns).reset_index()


def _get_zernike_coefficient(zernike_result: zp.analyses.base.AttrDict, coefficient: int) -> float:
    return zernike_result.Data.Coefficients.loc["Z" + str(coefficient)].Value


def _zernike_data_to_refraction(
    zernike_data: zp.analyses.base.AttrDict,
    pupil_data: zp.functions.lde.PupilData,
    wavelength: float,
    *,
    use_higher_order_aberrations: bool = True,
) -> FourierPowerVectorRefraction:
    z4 = _get_zernike_coefficient(zernike_data, 4) * wavelength * 4 * np.sqrt(3)
    z11 = _get_zernike_coefficient(zernike_data, 11) * wavelength * 12 * np.sqrt(5)
    z22 = _get_zernike_coefficient(zernike_data, 22) * wavelength * 24 * np.sqrt(7)
    z37 = _get_zernike_coefficient(zernike_data, 37) * wavelength * 40 * np.sqrt(9)

    z6 = _get_zernike_coefficient(zernike_data, 6) * wavelength * 2 * np.sqrt(6)
    z12 = _get_zernike_coefficient(zernike_data, 12) * wavelength * 6 * np.sqrt(10)
    z24 = _get_zernike_coefficient(zernike_data, 24) * wavelength * 12 * np.sqrt(14)
    z38 = _get_zernike_coefficient(zernike_data, 38) * wavelength * 60 * np.sqrt(2)

    z5 = _get_zernike_coefficient(zernike_data, 5) * wavelength * 2 * np.sqrt(6)
    z13 = _get_zernike_coefficient(zernike_data, 13) * wavelength * 6 * np.sqrt(10)
    z23 = _get_zernike_coefficient(zernike_data, 23) * wavelength * 12 * np.sqrt(14)
    z39 = _get_zernike_coefficient(zernike_data, 39) * wavelength * 60 * np.sqrt(2)

    exit_pupil_radius = pupil_data.ExitPupilDiameter / 2

    if use_higher_order_aberrations:
        return FourierPowerVectorRefraction(
            M=(-z4 + z11 - z22 + z37) / (exit_pupil_radius**2),
            J0=(-z6 + z12 - z24 + z38) / (exit_pupil_radius**2),
            J45=(-z5 + z13 - z23 + z39) / (exit_pupil_radius**2),
        )

    return FourierPowerVectorRefraction(
        M=(-z4) / (exit_pupil_radius**2),
        J0=(-z6) / (exit_pupil_radius**2),
        J45=(-z5) / (exit_pupil_radius**2),
    )


class OpticStudioAnalysis(BaseAnalysis):
    """
    Analyses for the OpticStudio backend.
    """

    def __init__(self, backend: OpticStudioBackend):
        self._backend = backend
        self._oss = backend.oss

    def cardinal_points(
        self, surface_1: int | None = None, surface_2: int | None = None
    ) -> tuple[CardinalPointsResult, zp.analyses.base.AnalysisResult]:
        """
        Get the cardinal points of the system between `surface_1` and `surface_2`.

        Parameters
        ----------
        surface_1 : int | None, optional
            The first surface to be used in the analysis. If `None`, the first surface in the system will be used.
            Defaults to `None`.
        surface_2 : int | None, optional
            The second surface to be used in the analysis. If `None`, the last surface in the system will be used.
            Defaults to `None`.

        Returns
        -------
        CardinalPointsResult
            The cardinal points of the system.

        Raises
        ------
        ValueError
            If `surface_1` or `surface_2` are not between 1 and the number of surfaces in the system, or if `surface_1`
            is greater than or equal to `surface_2`.
        """
        surface_1 = surface_1 or 1
        surface_2 = surface_2 or self._oss.LDE.NumberOfSurfaces - 1

        if surface_1 < 1 or surface_2 > self._oss.LDE.NumberOfSurfaces - 1:
            raise ValueError("surface_1 and surface_2 must be between 1 and the number of surfaces in the system.")

        if surface_1 >= surface_2:
            raise ValueError("surface_1 must be less than surface_2.")

        cardinal_points_result = zp.analyses.reports.cardinal_points(
            self._oss,
            surface_1=surface_1,
            surface_2=surface_2,
        )

        return _build_cardinal_points_result(cardinal_points_result), cardinal_points_result

    def raytrace(
        self,
        coordinates: Iterable[tuple[float, float]],
        wavelengths: Iterable[float] = (0.543,),
        field_type: Literal["angle", "object_height"] = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> tuple[DataFrame, list[zp.analyses.base.AnalysisResult]]:
        """
        Perform a ray trace analysis using the given parameters.
        The ray trace is performed for each wavelength and field in the system, using the Single Ray Trace analysis
        in OpticStudio.

        The analysis returns a Dataframe with the following columns:

        - field: The field coordinates for the ray trace.
        - wavelength: The wavelength used in the ray trace.
        - surface: The surface number in the system.
        - comment: The comment for the surface.
        - x: The X-coordinate of the ray trace.
        - y: The Y-coordinate of the ray trace.
        - z: The Z-coordinate of the ray trace.

        Parameters
        ----------
        coordinates : Iterable[tuple[float, float]]
            An iterable of tuples representing the coordinates for the ray trace.
            If `field_type` is "angle", the coordinates should be the angles along the (X, Y) axes in degrees.
            If `field_type` is "object_height", the coordinates should be the object heights along the
            (X, Y) axes in mm.
        wavelengths : Iterable[float], optional
            An iterable of wavelengths to be used in the ray trace. Defaults to (0.543,).
        field_type : Literal["angle", "object_height"], optional
            The type of field to be used in the ray trace. Can be either "angle" or "object_height". Defaults to
            "angle".
        pupil : tuple[float, float], optional
            A tuple representing the pupil coordinates for the ray trace. Defaults to (0, 0).

        Returns
        -------
        DataFrame
            A pandas DataFrame containing the results of the ray trace analysis.
        """
        self._backend.set_fields(coordinates, field_type=field_type)
        self._backend.set_wavelengths(wavelengths)

        raytrace_results = []

        for wavelength_number, wavelength in _iter_wavelengths(self._backend.oss):
            for field_number, field in _iter_fields(self._backend.oss):
                raytrace_result = zp.analyses.raysandspots.single_ray_trace(
                    self._backend.oss,
                    px=pupil[0],
                    py=pupil[1],
                    field=field_number,
                    wavelength=wavelength_number,
                    global_coordinates=True,
                ).Data.RealRayTraceData

                raytrace_result.insert(0, "Field", [(field.X, field.Y)] * len(raytrace_result))
                raytrace_result.insert(0, "Wavelength", wavelength)

                raytrace_results.append(raytrace_result)

        return _build_raytrace_result(raytrace_results), raytrace_results

    def zernike_standard_coefficients(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
        field_type: Literal["angle", "object_height"] = "angle",
        sampling: str = "512x512",
        maximum_term: int = 45,
    ) -> tuple[zp.analyses.base.AttrDict, zp.analyses.base.AnalysisResult]:
        """
        Calculates the Zernike standard coefficients at the retina surface.

        Parameters
        ----------
        field_coordinate : tuple[float, float] | None, optional
            The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
            Defaults to `None`.
        wavelength : float | None, optional
            The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
            Defaults to `None`.
        field_type : Literal["angle", "object_height"], optional
            The type of field to be used when setting the field coordinate. This parameter is only used when
            `field_coordinate` is specified. Defaults to "angle".
        sampling : str, optional
            The sampling for the Zernike calculation. Defaults to "512x512".
        maximum_term : int, optional
            The maximum term for the Zernike calculation. Defaults to 45.

        Returns
        -------
        AttrDict
            ZOSPy Zernike standard coefficients analysis output.
        """
        wavelength_number = 1 if wavelength is None else self._backend.get_wavelength_number(wavelength)

        if wavelength_number is None:
            self._backend.set_wavelengths([wavelength])
            wavelength_number = 1

        if field_coordinate is not None:
            self._backend.set_fields([field_coordinate], field_type=field_type)

        zernike_result = zp.analyses.wavefront.zernike_standard_coefficients(
            self._oss,
            sampling=sampling,
            maximum_term=maximum_term,
            wavelength=wavelength_number,
            field=1,
            reference_opd_to_vertex=False,
            surface="Image",
            sx=0.0,
            sy=0.0,
            sr=1.0,
        )

        return zernike_result.Data, zernike_result

    def refraction(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
        pupil_diameter: float | None = None,
        field_type: Literal["angle", "object_height"] = "angle",
        *,
        use_higher_order_aberrations: bool = True,
    ) -> tuple[FourierPowerVectorRefraction, zp.analyses.base.AnalysisResult]:
        """Calculates the ocular refraction.

        The ocular refraction is calculated from Zernike standard coefficients and represented in Fourier power
        vector form.

        Parameters
        ----------
        use_higher_order_aberrations : bool, optional
            If `True`, higher-order aberrations are used in the calculation. Defaults to `True`.
        field_coordinate : tuple[float, float], optional
            The field coordinate for the Zernike calculation. When `None`, the first field in OpticStudio is used.
            Defaults to `None`.
        wavelength : float, optional
            The wavelength for the Zernike calculation. When `None`, the first wavelength in OpticStudio is used.
            Defaults to `None`.
        pupil_diameter : float, optional
            The diameter of the pupil for the refraction calculation. Defaults to the pupil diameter configured in the
            model.
        field_type : Literal["angle", "object_height"], optional
            The type of field to be used when setting the field coordinate. This parameter is only used when
            `field_coordinate` is specified. Defaults to "angle".

        Returns
        -------
         FourierPowerVectorRefraction
              The ocular refraction in Fourier power vector form.
        """
        # Get the wavelength from OpticStudio if not specified
        wavelength = self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength if wavelength is None else wavelength

        # Temporarily change the pupil diameter
        old_pupil_semi_diameter = None
        if pupil_diameter is not None:
            old_pupil_semi_diameter = self._backend.model.pupil.semi_diameter
            self._backend.model.pupil.semi_diameter = pupil_diameter / 2

        pupil_data = zp.functions.lde.get_pupil(self._oss)
        _, zernike_standard_coefficients = self.zernike_standard_coefficients(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
        )

        if old_pupil_semi_diameter is not None:
            self._backend.model.pupil.semi_diameter = old_pupil_semi_diameter

        return _zernike_data_to_refraction(
            zernike_standard_coefficients,
            pupil_data,
            wavelength,
            use_higher_order_aberrations=use_higher_order_aberrations,
        ), zernike_standard_coefficients
