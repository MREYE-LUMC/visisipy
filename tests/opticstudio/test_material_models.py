from __future__ import annotations

import numpy as np
import pytest
import zospy as zp

from visisipy.models.materials import NavarroMaterials

pytestmark = [pytest.mark.needs_opticstudio]


@pytest.fixture
def opticstudio_model(oss):
    """Simple optical system with 3 parallel surfaces: a stop, refracting interface, and image surface.
    All surfaces except the refracting interface and image surface have a refractive index of 1.0.
    The refractive index of the image surface is equal to that of the refracting interface, to prevent reflection.

    When a single ray trace is performed, the angle of incidence at the image surface is equal to the angle of
    refraction at the refracting interface. This allows to calculate the refractive index of the refracting interface
    from the angle of incidence at the refracting interface and the angle of incidence at the image surface.
    """
    oss.SystemData.Aperture.ApertureType = zp.constants.SystemData.ZemaxApertureType.FloatByStopSize

    oss.SystemData.Fields.SetFieldType(zp.constants.SystemData.FieldType.Angle)
    oss.SystemData.Fields.GetField(1).Y = 20

    obj = oss.LDE.GetSurfaceAt(0)
    obj.Comment = "object"
    obj.Thickness = float("inf")
    zp.solvers.material_model(
        obj.MaterialCell,
        refractive_index=1.0,
        abbe_number=0.0,
        partial_dispersion=0.0,
    )

    stop = oss.LDE.GetSurfaceAt(1)
    stop.Comment = "stop"
    stop.Thickness = 5.0
    stop.SemiDiameter = 1.0
    zp.solvers.material_model(stop.MaterialCell, refractive_index=1.0, abbe_number=0.0, partial_dispersion=0.0)

    interface = oss.LDE.InsertNewSurfaceAt(2)
    interface.Comment = "interface"
    interface.Thickness = 5.0
    zp.solvers.material_model(
        interface.MaterialCell,
        refractive_index=1.5,
        abbe_number=0.0,
        partial_dispersion=0.0,
    )

    image = oss.LDE.GetSurfaceAt(3)
    image.Comment = "image"
    zp.solvers.material_model(
        image.MaterialCell,
        refractive_index=1.5,
        abbe_number=0.0,
        partial_dispersion=0.0,
    )

    return oss


def _set_material_model(material_cell, material_model):
    zp.solvers.material_model(
        material_cell,
        refractive_index=material_model.refractive_index,
        abbe_number=material_model.abbe_number,
        partial_dispersion=material_model.partial_dispersion,
    )


@pytest.mark.parametrize(
    "material_model,wavelength,expected_index",
    [
        # Cornea
        (NavarroMaterials().cornea, 0.458, 1.3828),
        (NavarroMaterials().cornea, 0.543, 1.3777),
        (NavarroMaterials().cornea, 0.5893, 1.376),
        (NavarroMaterials().cornea, 0.6328, 1.3747),
        # Aqueous
        (NavarroMaterials().aqueous, 0.458, 1.3445),
        (NavarroMaterials().aqueous, 0.543, 1.3391),
        (NavarroMaterials().aqueous, 0.5893, 1.3374),
        (NavarroMaterials().aqueous, 0.6328, 1.336),
        # Lens
        (NavarroMaterials().lens, 0.458, 1.4292),
        (NavarroMaterials().lens, 0.543, 1.4222),
        (NavarroMaterials().lens, 0.5893, 1.42),
        (NavarroMaterials().lens, 0.6328, 1.4183),
        # Vitreous
        (NavarroMaterials().vitreous, 0.458, 1.3428),
        (NavarroMaterials().vitreous, 0.543, 1.3377),
        (NavarroMaterials().vitreous, 0.5893, 1.336),
        (NavarroMaterials().vitreous, 0.6328, 1.3347),
    ],
)
def test_material_model_refractive_index(opticstudio_model, material_model, wavelength, expected_index):
    # Set wavelength
    opticstudio_model.SystemData.Wavelengths.GetWavelength(1).Wavelength = wavelength

    # Set material model of test surface and image surface to the material model
    _set_material_model(opticstudio_model.LDE.GetSurfaceAt(2).MaterialCell, material_model)
    _set_material_model(opticstudio_model.LDE.GetSurfaceAt(3).MaterialCell, material_model)

    # Get angle of incidence at image surface. As the refracting surface and the image surface are parallel, the angle
    # of incidence at the image surface is equal to the angle of refraction at the refracting surface.
    raytrace_result = zp.analyses.raysandspots.SingleRayTrace(
        hx=0, hy=1, px=0, py=1, wavelength=1, global_coordinates=True
    ).run(opticstudio_model)
    angle = raytrace_result.data.real_ray_trace_data.loc[3]["Angle in"]
    refractive_index = np.sin(np.deg2rad(20)) / np.sin(np.deg2rad(angle))

    assert refractive_index == pytest.approx(expected_index, rel=1e-3)
