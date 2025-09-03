"""PSF analyses for OpticStudio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import zospy as zp
from optiland.distribution import HexagonalDistribution

from visisipy.opticstudio.analysis.helpers import set_field, set_wavelength
from visisipy.types import FieldCoordinate, FieldType, SampleSize

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from zospy.analyses.psf.huygens_psf import HuygensPSFResult
    from zospy.zpcore import OpticStudioSystem

    from visisipy.opticstudio import OpticStudioBackend


__all__ = ("fft_psf", "huygens_psf", "strehl_ratio")


def fft_psf(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the FFT Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in OpticStudio is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in OpticStudio is used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The field type. Either "angle" or "object_height". Defaults to "angle". This parameter is only used if
        `field_coordinate` is not `None`.
    sampling : SampleSize | str | int, optional
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.
    """

    if not isinstance(sampling, SampleSize):
        sampling = SampleSize(sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type)

    psf_result = zp.analyses.psf.FFTPSF(
        sampling=str(sampling),
        display=str(2 * sampling),
        wavelength=wavelength_number,
        field=field_number,
        psf_type=zp.constants.Analysis.Settings.Psf.FftPsfType.Linear,
        surface="Image",
        normalize=False,
    ).run(backend.get_oss())

    if psf_result.data is None:
        raise ValueError("Failed to run FFT PSF analysis.")

    return psf_result.data, psf_result.data


class RayTraceResult(NamedTuple):
    """Result of a ray trace in OpticStudio."""

    ray_number: int
    error_code: int
    vignette_code: int
    x: float
    y: float
    z: float
    l: float  # noqa: E741
    m: float
    n: float
    l2: float
    m2: float
    n2: float
    opd: float
    intensity: float


def _opticstudio_batch_raytrace(
    oss: OpticStudioSystem,
    wavelength_number: int,
    h_x: float,
    h_y: float,
    p_x: NDArray,
    p_y: NDArray,
) -> list[RayTraceResult]:
    """Perform a batch ray trace in OpticStudio.

    Uses the BatchRayTrace tool with NormUnpol rays to trace rays at specified pupil coordinates.

    Parameters
    ----------
    oss : OpticStudioSystem
        OpticStudio system to perform the ray trace in.
    wavelength_number : int
        Wavelength number to use for the ray trace.
    h_x : float
        Normalized x-coordinate of the field.
    h_y : float
        Normalized y-coordinate of the field.
    p_x : NDArray
        Array of x-coordinates in the pupil to trace rays.
    p_y : NDArray
        Array of y-coordinates in the pupil to trace rays.

    Returns
    -------
    list[RayTraceResult]
        List of RayTraceResult objects containing the results of the ray trace.
    """
    if len(p_x) != len(p_y):
        raise ValueError("p_x and p_y must have the same length")

    p_x, p_y = np.array(p_x), np.array(p_y)

    if oss.Tools.CurrentTool is not None:
        oss.Tools.CurrentTool.Close()

    batch_raytrace = oss.Tools.OpenBatchRayTrace()
    norm_unpol = batch_raytrace.CreateNormUnpol(
        MaxRays=p_x.size,
        rayType=zp.constants.Tools.RayTrace.RaysType.Real,
        toSurface=oss.LDE.NumberOfSurfaces,
    )

    for x, y in zip(p_x, p_y, strict=False):
        norm_unpol.AddRay(
            waveNumber=wavelength_number,
            Hx=h_x,
            Hy=h_y,
            Px=float(x),  # Make sure Python floats are passed
            Py=float(y),
            calcOPD=zp.constants.Tools.RayTrace.OPDMode.None_,
        )

    batch_raytrace.RunAndWaitForCompletion()
    norm_unpol.StartReadingResults()

    result = []

    for _ in range(norm_unpol.NumberOfRays):
        success, *ray_trace_result = norm_unpol.ReadNextResult()
        if not success:
            raise RuntimeError("Failed to read ray trace result")

        result.append(RayTraceResult(*ray_trace_result))

    norm_unpol.ClearData()
    batch_raytrace.Close()

    return result


def _get_huygens_psf_extent(oss: OpticStudioSystem, field: int = 1, wavelength_number: int = 1) -> float:
    """Calculate the extent for the Huygens PSF based on the geometric spot size or ideal extent.

    Parameters
    ----------
    oss : OpticStudioSystem
        OpticStudio system to use for the calculation.
    field : int, optional
        Field number to use for the calculation. Defaults to 1.
    wavelength_number : int, optional
        Wavelength number to use for the calculation. Defaults to 1.

    Returns
    -------
    float
        The extent for the Huygens PSF in μm.
    """
    fields = [
        [oss.SystemData.Fields.GetField(i).X, oss.SystemData.Fields.GetField(i).Y]
        for i in range(1, oss.SystemData.Fields.NumberOfFields + 1)
    ]
    max_field = np.max(np.linalg.norm(fields, axis=1))

    oss_field = oss.SystemData.Fields.GetField(field)
    h_x, h_y = (oss_field.X / max_field, oss_field.Y / max_field) if max_field > 0 else (0, 0)

    pupil_distribution = HexagonalDistribution()
    pupil_distribution.generate_points(6)

    ray_trace_results = _opticstudio_batch_raytrace(
        oss,
        wavelength_number=wavelength_number,
        h_x=h_x,
        h_y=h_y,
        p_x=pupil_distribution.x,
        p_y=pupil_distribution.y,
    )

    rays_x = np.array([r.x for r in ray_trace_results])
    rays_y = np.array([r.y for r in ray_trace_results])
    center_x = np.mean(rays_x)
    center_y = np.mean(rays_y)

    geometric_spot_size = np.hypot(rays_x - center_x, rays_y - center_y).max() * 1000  # Convert to μm

    system_data = zp.analyses.reports.SystemData().run(oss)
    wavelength = oss.SystemData.Wavelengths.GetWavelength(wavelength_number).Wavelength

    ideal_extent = (
        5
        * system_data.data.general_lens_data.working_f_number  # effective F-number
        * 1.22
        * wavelength
    )

    return 2 * max(geometric_spot_size, ideal_extent)


def huygens_psf(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    pupil_sampling: SampleSize | str | int = 128,
    image_sampling: SampleSize | str | int = 128,
) -> tuple[pd.DataFrame, HuygensPSFResult]:
    """Calculate the Huygens Point Spread Function (PSF) at the retina surface.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : tuple[float, float], optional
        The field coordinate (x, y) in mm. If `None`, the first field in OpticStudio is used. Defaults to `None`.
    wavelength : float, optional
        The wavelength in μm. If `None`, the first wavelength in OpticStudio is used. Defaults to `None`.
    field_type : Literal["angle", "object_height"], optional
        The field type. Either "angle" or "object_height". Defaults to "angle". This parameter is only used if
        `field_coordinate` is not `None`.
    pupil_sampling : SampleSize | str | int, optional
        The size of the ray grid used to sample the pupil, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.
    image_sampling : SampleSize | str | int, optional
        The size of the PSF grid, either string (e.g. '32x32') or int (e.g. 32). Defaults to 128.

    Returns
    -------
    DataFrame
        The PSF data as a pandas DataFrame.
    HuygensPSFData
        The Huygens PSF result from OpticStudio.
    """

    if not isinstance(pupil_sampling, SampleSize):
        pupil_sampling = SampleSize(pupil_sampling)

    if not isinstance(image_sampling, SampleSize):
        image_sampling = SampleSize(image_sampling)

    wavelength_number = set_wavelength(backend, wavelength)
    field_number = set_field(backend, field_coordinate, field_type)
    extent = _get_huygens_psf_extent(backend.get_oss(), field=field_number, wavelength_number=wavelength_number)
    image_delta = extent / int(image_sampling)

    psf_result = zp.analyses.psf.HuygensPSFAndStrehlRatio(
        pupil_sampling=str(pupil_sampling),
        image_sampling=str(image_sampling),
        image_delta=image_delta,
        rotation=0,
        wavelength=wavelength_number,
        field=field_number,
        psf_type="Linear",
        show_as="Surface",
        use_polarization=False,
        use_centroid=False,
        normalize=False,
    ).run(backend.get_oss())

    return psf_result.data.psf, psf_result.data


def strehl_ratio(
    backend: type[OpticStudioBackend],
    field_coordinate: FieldCoordinate | None = None,
    wavelength: float | None = None,
    field_type: FieldType = "angle",
    sampling: SampleSize | str | int = 128,
    psf_type: Literal["fft", "huygens"] = "huygens",
) -> tuple[float, HuygensPSFResult]:
    """Calculate the Strehl ratio of the optical system.

    The Strehl ratio is calculated from the point spread function. Which PSF is used depends on the `psf_type` parameter.

    Parameters
    ----------
    backend : type[OpticStudioBackend]
        Reference to the OpticStudio backend.
    field_coordinate : FieldCoordinate | None
        The field coordinate at which the Strehl ratio is calculated. If `None`, the first field coordinate in
        OpticStudio is used.
    wavelength : float | None
        The wavelength at which the Strehl ratio is calculated. If `None`, the first wavelength in OpticStudio is used.
    field_type : FieldType
        The field type to be used in the analysis. Can be either "angle" or "object_height". Defaults to "angle".
        This parameter is only used when `field_coordinate` is specified.
    sampling : SampleSize | str | int
        The size of the ray grid used to sample the pupil. Can be an integer or a string in the format "NxN", where N
        is an integer. Defaults to 128.
    psf_type : Literal["fft", "huygens"]
        The type of PSF to be used for the Strehl ratio calculation. Can be either "fft" or "huygens". Defaults to "huygens";
        OpticStudio's FFT PSF does not support calculating the Strehl ratio, so only "huygens" is supported.

    Returns
    -------
    float
        The Strehl ratio of the optical system at the specified field coordinate and wavelength.
    HuygensPSFResult
        The PSF object used to calculate the Strehl ratio. The type of the object depends on the `psf_type` parameter.
    """
    if psf_type == "fft":
        raise NotImplementedError("OpticStudio does not support obtaining the Strehl ratio from the FFT PSF.")

    if psf_type == "huygens":
        _, psf = huygens_psf(
            backend=backend,
            field_coordinate=field_coordinate,
            wavelength=wavelength,
            field_type=field_type,
            pupil_sampling=sampling,
            image_sampling=sampling,
        )

        return psf.strehl_ratio, psf

    raise NotImplementedError(f"PSF type '{psf_type}' is not implemented. Only 'huygens' is supported.")
