from __future__ import annotations

import visisipy
from visisipy.backend import BackendSettings
from visisipy.opticstudio import OpticStudioBackend
from visisipy.optiland import OptilandBackend

model = visisipy.EyeModel()

settings = BackendSettings(
    field_type="angle",
    fields=[(0, 0), (0, 10), (20, 0), (30, 40)],
    wavelengths=[0.400, 0.543, 0.550, 0.650],
    aperture_type="entrance_pupil_diameter",
    aperture_value=4.321,
)

OpticStudioBackend.initialize(**settings, ray_aiming="off")
OptilandBackend.initialize(**settings)

OpticStudioBackend.build_model(model)
OpticStudioBackend.save_model("navarro_eye.zmx")

OptilandBackend.build_model(model)
OptilandBackend.save_model("navarro_eye.json")
