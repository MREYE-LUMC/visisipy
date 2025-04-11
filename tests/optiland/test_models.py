from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from visisipy.optiland import OptilandEye

if TYPE_CHECKING:
    from optiland.optic import Optic

    from visisipy import EyeModel


class TestOptilandEye:
    def test_init(self, eye_model):
        opticstudio_eye = OptilandEye(eye_model)

        assert opticstudio_eye.eye_model == eye_model

    def test_build_cornea_front(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optic.surface_group.get_thickness(1) == pytest.approx(
            eye_model.geometry.cornea_front.thickness
        )
        assert optic.surface_group.radii[1] == pytest.approx(
            eye_model.geometry.cornea_front.radius
        )
        assert optic.surface_group.conic[1] == pytest.approx(
            eye_model.geometry.cornea_front.asphericity
        )

    def test_build_cornea_back(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optic.surface_group.get_thickness(2) == pytest.approx(
            eye_model.geometry.cornea_back.thickness
        )
        assert optic.surface_group.radii[2] == pytest.approx(
            eye_model.geometry.cornea_back.radius
        )
        assert optic.surface_group.conic[2] == pytest.approx(
            eye_model.geometry.cornea_back.asphericity
        )

    def test_build_pupil(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        # assert optic.surface_group.surfaces[3].semi_aperture
        assert optic.surface_group.stop_index == 3
        assert optic.surface_group.get_thickness(3) == 0
        assert optic.surface_group.radii[3] == float("inf")

    def test_build_lens_front(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optic.surface_group.get_thickness(4) == pytest.approx(
            eye_model.geometry.lens_front.thickness
        )
        assert optic.surface_group.radii[4] == pytest.approx(
            eye_model.geometry.lens_front.radius
        )
        assert optic.surface_group.conic[4] == pytest.approx(
            eye_model.geometry.lens_front.asphericity
        )

    def test_build_lens_back(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optic.surface_group.get_thickness(5) == pytest.approx(
            eye_model.geometry.lens_back.thickness
        )
        assert optic.surface_group.radii[5] == pytest.approx(
            eye_model.geometry.lens_back.radius
        )
        assert optic.surface_group.conic[5] == pytest.approx(
            eye_model.geometry.lens_back.asphericity
        )

    def test_build_retina(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optic.surface_group.radii[6] == pytest.approx(
            eye_model.geometry.retina.radius
        )
        assert optic.surface_group.conic[6] == pytest.approx(
            eye_model.geometry.retina.asphericity
        )
        assert optic.image_surface == optiland_eye.retina.surface

    def test_build_pupil_not_stop_raises_valueerror(
        self, optic: Optic, eye_model: EyeModel
    ):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.lens_front._is_stop = True

        with pytest.raises(
            ValueError, match="The pupil is not located at the stop position"
        ):
            optiland_eye.build(optic)

    def test_build_retina_not_image_raises_valueerror(
        self, optic: Optic, eye_model: EyeModel
    ):
        optiland_eye = OptilandEye(eye_model)

        # Surfaces are inserted before an existing surface if a surface is already present at the specified index.
        optic.add_surface(index=0, comment="Dummy 1")
        optic.add_surface(index=0, comment="Dummy 2")

        with pytest.raises(
            ValueError, match="The retina is not located at the image position"
        ):
            optiland_eye.build(optic)

    def test_get_cornea_front_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.cornea_front.comment == "cornea front"

    def test_get_cornea_back_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.cornea_back.comment == "cornea back / aqueous"

    def test_get_pupil_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.pupil.comment == "pupil"

    def test_get_lens_front_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.lens_front.comment == "lens front"

    def test_get_lens_back_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.lens_back.comment == "lens back / vitreous"

    def test_get_retina_comment(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        assert optiland_eye.retina.comment == "retina"

    @pytest.mark.parametrize("build", [True, False])
    def test_surfaces(self, build: bool, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)

        if build:
            optiland_eye.build(optic)

        assert optiland_eye.surfaces == {
            "cornea_front": optiland_eye.cornea_front,
            "cornea_back": optiland_eye.cornea_back,
            "pupil": optiland_eye.pupil,
            "lens_front": optiland_eye.lens_front,
            "lens_back": optiland_eye.lens_back,
            "retina": optiland_eye.retina,
        }

    def test_update_surfaces(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        optiland_eye.update_surfaces("comment", "new comment")

        assert optiland_eye.cornea_front.comment == "new comment"
        assert optiland_eye.cornea_back.comment == "new comment"
        assert optiland_eye.pupil.comment == "new comment"
        assert optiland_eye.lens_front.comment == "new comment"
        assert optiland_eye.lens_back.comment == "new comment"
        assert optiland_eye.retina.comment == "new comment"

    def test_update_surfaces_subset(self, optic: Optic, eye_model: EyeModel):
        optiland_eye = OptilandEye(eye_model)
        optiland_eye.build(optic)

        optiland_eye.update_surfaces(
            "comment", "new comment", surface_names=["cornea_front", "lens_back"]
        )

        assert optiland_eye.cornea_front.comment == "new comment"
        assert optiland_eye.cornea_back.comment != "new comment"
        assert optiland_eye.pupil.comment != "new comment"
        assert optiland_eye.lens_front.comment != "new comment"
        assert optiland_eye.lens_back.comment == "new comment"
        assert optiland_eye.retina.comment != "new comment"

