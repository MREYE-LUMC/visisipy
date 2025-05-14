from __future__ import annotations

import numpy as np
import pytest
from optiland.materials import AbbeMaterial, IdealMaterial
from optiland.optic import Optic

from visisipy.models.materials import NavarroMaterials


def build_model(wavelength: float, n: float, abbe: float) -> Optic:
    """Simple optical system with 2 parallel surfaces: a stop / refracting interface, and image surface.
    All surfaces except the refracting interface and image surface have a refractive index of 1.0.
    The refractive index of the image surface is equal to that of the refracting interface, to prevent reflection.

    When a single ray trace is performed, the angle of incidence at the image surface is equal to the angle of
    refraction at the refracting interface. This allows to calculate the refractive index of the refracting interface
    from the angle of incidence at the refracting interface and the angle of incidence at the image surface.
    """
    system = Optic()

    system.set_field_type("angle")
    system.add_field(x=0, y=20)
    system.set_aperture("EPD", 1.0)
    system.add_wavelength(wavelength)

    air = IdealMaterial(n=1)
    test_material = AbbeMaterial(n=n, abbe=abbe)

    system.add_surface(index=0, comment="object", thickness=float("inf"), material=air)
    system.add_surface(index=1, comment="stop", is_stop=True, thickness=5.0, material=test_material)
    system.add_surface(index=2, comment="image", thickness=0.0, material=test_material)

    return system


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
def test_material_model_refractive_index(material_model, wavelength, expected_index):
    system = build_model(wavelength, material_model.refractive_index, material_model.abbe_number)

    system.trace_generic(0, 1, 0, 0, wavelength=wavelength)
    trace_z, trace_y = system.surface_group.z, system.surface_group.y
    sin_angle_out = trace_y[2] / np.sqrt(trace_z[2] ** 2 + trace_y[2] ** 2)  # Assuming the ray passes through (0, 0)
    refractive_index = np.sin(np.deg2rad(20)) / sin_angle_out

    assert (trace_z[1], trace_y[1]) == (0, 0), "Ray must pass through the pupil center"
    assert refractive_index == pytest.approx(expected_index, rel=1e-4)
