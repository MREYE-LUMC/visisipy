from __future__ import annotations

from typing import Iterable, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import zospy as zp
from pandas import DataFrame

from visisipy._backend import BaseAnalysis
from visisipy.analysis.refraction import FourierPowerVectorRefraction

if TYPE_CHECKING:
    from zospy.api import _ZOSAPI
    from zospy.zpcore import OpticStudioSystem
    from visisipy.opticstudio.backend import OpticStudioBackend


def _iter_fields(oss: OpticStudioSystem) -> tuple[int, _ZOSAPI.SystemData.IField]:
    for i in range(oss.SystemData.Fields.NumberOfFields):
        field = oss.SystemData.Fields.GetField(i + 1)
        yield field.FieldNumber, field


def _iter_wavelengths(oss: OpticStudioSystem) -> tuple[int, float]:
    for i in range(oss.SystemData.Wavelengths.NumberOfWavelengths):
        yield i + 1, oss.SystemData.Wavelengths.GetWavelength(i + 1)


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

    df = (
        pd.concat(raytrace_results)[columns.keys()]
        .rename(columns=columns)
        .reset_index()
    )

    return df


def _get_zernike_coefficient(
    zernike_result: zp.analyses.base.AttrDict, coefficient: int
) -> float:
    return zernike_result.Data.Coefficients.loc["Z" + str(coefficient)].Value


def _zernike_data_to_refraction(
    zernike_data: zp.analyses.base.AttrDict,
    pupil_data: zp.functions.lde.PupilData,
    wavelength: float,
    use_higher_order_aberrations: bool = True,
) -> FourierPowerVectorRefraction:
    Z4 = _get_zernike_coefficient(zernike_data, 4) * wavelength * 4 * np.sqrt(3)
    Z11 = _get_zernike_coefficient(zernike_data, 11) * wavelength * 12 * np.sqrt(5)
    Z22 = _get_zernike_coefficient(zernike_data, 22) * wavelength * 24 * np.sqrt(7)
    Z37 = _get_zernike_coefficient(zernike_data, 37) * wavelength * 40 * np.sqrt(9)

    Z6 = _get_zernike_coefficient(zernike_data, 6) * wavelength * 2 * np.sqrt(6)
    Z12 = _get_zernike_coefficient(zernike_data, 12) * wavelength * 6 * np.sqrt(10)
    Z24 = _get_zernike_coefficient(zernike_data, 24) * wavelength * 12 * np.sqrt(14)
    Z38 = _get_zernike_coefficient(zernike_data, 38) * wavelength * 60 * np.sqrt(2)

    Z5 = _get_zernike_coefficient(zernike_data, 5) * wavelength * 2 * np.sqrt(6)
    Z13 = _get_zernike_coefficient(zernike_data, 13) * wavelength * 6 * np.sqrt(10)
    Z23 = _get_zernike_coefficient(zernike_data, 23) * wavelength * 12 * np.sqrt(14)
    Z39 = _get_zernike_coefficient(zernike_data, 39) * wavelength * 60 * np.sqrt(2)

    exit_pupil_radius = pupil_data.ExitPupilDiameter / 2

    if use_higher_order_aberrations:
        return FourierPowerVectorRefraction(
            M=(-Z4 + Z11 - Z22 + Z37) / (exit_pupil_radius**2),
            J0=(-Z6 + Z12 - Z24 + Z38) / (exit_pupil_radius**2),
            J45=(-Z5 + Z13 - Z23 + Z39) / (exit_pupil_radius**2),
        )

    return FourierPowerVectorRefraction(
        M=(-Z4) / (exit_pupil_radius**2),
        J0=(-Z6) / (exit_pupil_radius**2),
        J45=(-Z5) / (exit_pupil_radius**2),
    )


class OpticStudioAnalysis(BaseAnalysis):
    """
    Analyses for the OpticStudio backend.
    """

    def __init__(self, backend: OpticStudioBackend):
        self._backend = backend
        self._oss = backend._oss

    def raytrace(
        self,
        coordinates: Iterable[tuple[float, float]],
        wavelengths: Iterable[float] = (0.543,),
        field_type: Literal["angle", "object_height"] = "angle",
        pupil: tuple[float, float] = (0, 0),
    ) -> DataFrame:
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

        for wavelength_number, wavelength in _iter_wavelengths(self._backend._oss):
            for field_number, field in _iter_fields(self._backend._oss):
                raytrace_result = zp.analyses.raysandspots.single_ray_trace(
                    self._backend._oss,
                    px=pupil[0],
                    py=pupil[1],
                    field=field_number,
                    wavelength=wavelength_number,
                    global_coordinates=True,
                ).Data.RealRayTraceData

                raytrace_result.insert(
                    0, "Field", [(field.X, field.Y)] * len(raytrace_result)
                )
                raytrace_result.insert(0, "Wavelength", wavelength)

                raytrace_results.append(raytrace_result)

        return _build_raytrace_result(raytrace_results)

    def zernike_standard_coefficients(
        self,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
        field_type: Literal["angle", "object_height"] = "angle",
        sampling: str = "512x512",
        maximum_term: int = 45,
    ) -> tuple[zp.analyses.base.AttrDict, zp.analyses.base.AttrDict]:
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
        wavelength_number = (
            1 if wavelength is None else self._backend.get_wavelength_number(wavelength)
        )

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

        return zernike_result, zernike_result

    def refraction(
        self,
        use_higher_order_aberrations: bool = True,
        field_coordinate: tuple[float, float] | None = None,
        wavelength: float | None = None,
        pupil_diameter: float | None = None,
        field_type: Literal["angle", "object_height"] = "angle",
    ) -> tuple[FourierPowerVectorRefraction, zp.analyses.base.AttrDict]:
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
        wavelength = (
            self._oss.SystemData.Wavelengths.GetWavelength(1).Wavelength
            if wavelength is None
            else wavelength
        )

        # Temporarily change the pupil diameter
        old_pupil_diameter = None
        if pupil_diameter is not None:
            old_pupil_diameter = self._backend._model.pupil.semi_diameter * 2
            self._backend._model.pupil.semi_diameter = pupil_diameter / 2

        pupil_data = zp.functions.lde.get_pupil(self._oss)
        zernike_standard_coefficients, _ = self.zernike_standard_coefficients(
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
        )

        if old_pupil_diameter is not None:
            self._backend._model.pupil.semi_diameter = old_pupil_diameter

        return _zernike_data_to_refraction(
            zernike_standard_coefficients,
            pupil_data,
            wavelength,
            use_higher_order_aberrations,
        ), zernike_standard_coefficients
