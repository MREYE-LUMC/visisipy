from __future__ import annotations

import pytest

from visisipy.opticstudio import OpticStudioEye

# pyright: reportOptionalMemberAccess=false

pytestmark = [pytest.mark.needs_opticstudio]


class TestOpticStudioEye:
    def test_init(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)

        assert opticstudio_eye.eye_model == eye_model

    def test_build_cornea_front(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.cornea_front.thickness == pytest.approx(eye_model.geometry.cornea_front.thickness)
        assert opticstudio_eye.cornea_front.radius == pytest.approx(eye_model.geometry.cornea_front.radius)
        assert opticstudio_eye.cornea_front.conic == pytest.approx(eye_model.geometry.cornea_front.asphericity)

    def test_build_cornea_back(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.cornea_back.thickness == pytest.approx(eye_model.geometry.cornea_back.thickness)
        assert opticstudio_eye.cornea_back.radius == pytest.approx(eye_model.geometry.cornea_back.radius)
        assert opticstudio_eye.cornea_back.conic == pytest.approx(eye_model.geometry.cornea_back.asphericity)

    def test_build_pupil(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.pupil.semi_diameter == pytest.approx(eye_model.geometry.pupil.semi_diameter)
        assert opticstudio_eye.pupil.is_stop

    def test_build_lens_front(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.lens_front.thickness == pytest.approx(eye_model.geometry.lens_front.thickness)
        assert opticstudio_eye.lens_front.radius == pytest.approx(eye_model.geometry.lens_front.radius)
        assert opticstudio_eye.lens_front.conic == pytest.approx(eye_model.geometry.lens_front.asphericity)

    def test_build_lens_back(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.lens_back.thickness == pytest.approx(eye_model.geometry.lens_back.thickness)
        assert opticstudio_eye.lens_back.radius == pytest.approx(eye_model.geometry.lens_back.radius)
        assert opticstudio_eye.lens_back.conic == pytest.approx(eye_model.geometry.lens_back.asphericity)

    def test_build_retina(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        assert opticstudio_eye.retina.radius == pytest.approx(eye_model.geometry.retina.radius)
        assert opticstudio_eye.retina.conic == pytest.approx(eye_model.geometry.retina.asphericity)
        assert opticstudio_eye.retina.surface.IsImage

    def test_build_pupil_not_stop_raises_valueerror(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.lens_front._is_stop = True

        with pytest.raises(ValueError, match="The pupil is not located at the stop position"):
            opticstudio_eye.build(oss)

    def test_build_retina_not_image_raises_valueerror(self, oss, eye_model):
        oss.LDE.InsertNewSurfaceAt(oss.LDE.NumberOfSurfaces)
        opticstudio_eye = OpticStudioEye(eye_model)

        with pytest.raises(ValueError, match="The retina is not located at the image position"):
            opticstudio_eye.build(oss)

    def test_build_start_from_index(self, oss, eye_model):
        oss.LDE.InsertNewSurfaceAt(1).Comment = "dummy"

        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss, start_from_index=1)

        assert opticstudio_eye.cornea_front.surface.SurfaceNumber == 2
        assert opticstudio_eye.cornea_back.surface.SurfaceNumber == 3
        assert opticstudio_eye.pupil.surface.SurfaceNumber == 4
        assert opticstudio_eye.lens_front.surface.SurfaceNumber == 5
        assert opticstudio_eye.lens_back.surface.SurfaceNumber == 6
        assert opticstudio_eye.retina.surface.SurfaceNumber == 7

    def test_build_start_from_index_after_stop_surface_raises_valueerror(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)

        with pytest.raises(ValueError, match="'start_from_index' must be smaller than the index of the stop surface"):
            opticstudio_eye.build(oss, start_from_index=1)

    def test_build_replace_existing(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        eye_model.geometry.cornea_front.thickness = 0.1
        eye_model.geometry.lens_front.radius = 12

        new_opticstudio_eye = OpticStudioEye(eye_model)
        new_opticstudio_eye.build(oss, replace_existing=True)

        assert oss.LDE.GetSurfaceAt(1).Thickness == 0.1
        assert oss.LDE.GetSurfaceAt(4).Radius == 12

    def test_build_object_distance(self, oss, eye_model):
        assert oss.LDE.GetSurfaceAt(0).Thickness == float("inf")

        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss, object_distance=1.0)

        assert oss.LDE.GetSurfaceAt(0).Thickness == 1.0

    def test_set_cornea_front_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.cornea_front.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(1).Comment == "new comment"

    def test_set_cornea_back_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.cornea_back.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(2).Comment == "new comment"

    def test_set_pupil_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.pupil.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(3).Comment == "new comment"

    def test_set_lens_front_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.lens_front.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(4).Comment == "new comment"

    def test_set_lens_back_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.lens_back.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(5).Comment == "new comment"

    def test_set_retina_property(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.retina.comment = "new comment"

        assert oss.LDE.GetSurfaceAt(6).Comment == "new comment"

    @pytest.mark.parametrize("build", [True, False])
    def test_surfaces(self, oss, eye_model, build):
        opticstudio_eye = OpticStudioEye(eye_model)

        if build:
            opticstudio_eye.build(oss)

        assert opticstudio_eye.surfaces == {
            "cornea_front": opticstudio_eye.cornea_front,
            "cornea_back": opticstudio_eye.cornea_back,
            "pupil": opticstudio_eye.pupil,
            "lens_front": opticstudio_eye.lens_front,
            "lens_back": opticstudio_eye.lens_back,
            "retina": opticstudio_eye.retina,
        }

    def test_update_surfaces(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.update_surfaces("comment", "new comment")

        assert opticstudio_eye.cornea_front.comment == "new comment"
        assert opticstudio_eye.cornea_back.comment == "new comment"
        assert opticstudio_eye.pupil.comment == "new comment"
        assert opticstudio_eye.lens_front.comment == "new comment"
        assert opticstudio_eye.lens_back.comment == "new comment"
        assert opticstudio_eye.retina.comment == "new comment"

    def test_update_surfaces_subset(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        opticstudio_eye.update_surfaces("comment", "new comment", surface_names=["cornea_front", "lens_back"])

        assert opticstudio_eye.cornea_front.comment == "new comment"
        assert opticstudio_eye.cornea_back.comment == "cornea back / aqueous"
        assert opticstudio_eye.pupil.comment == "pupil"
        assert opticstudio_eye.lens_front.comment == "lens front"
        assert opticstudio_eye.lens_back.comment == "new comment"
        assert opticstudio_eye.retina.comment == "retina"

    def test_relink_surfaces(self, oss, eye_model):
        opticstudio_eye = OpticStudioEye(eye_model)
        opticstudio_eye.build(oss)

        oss.LDE.InsertNewSurfaceAt(1).Comment = "dummy"

        assert opticstudio_eye.relink_surfaces(oss)
        assert opticstudio_eye.cornea_front.surface.SurfaceNumber == 2
        assert opticstudio_eye.cornea_front.comment == "cornea front"
        assert opticstudio_eye.cornea_back.surface.SurfaceNumber == 3
        assert opticstudio_eye.cornea_back.comment == "cornea back / aqueous"
        assert opticstudio_eye.pupil.surface.SurfaceNumber == 4
        assert opticstudio_eye.pupil.comment == "pupil"
        assert opticstudio_eye.lens_front.surface.SurfaceNumber == 5
        assert opticstudio_eye.lens_front.comment == "lens front"
        assert opticstudio_eye.lens_back.surface.SurfaceNumber == 6
        assert opticstudio_eye.lens_back.comment == "lens back / vitreous"
        assert opticstudio_eye.retina.surface.SurfaceNumber == 7
        assert opticstudio_eye.retina.comment == "retina"


def test_model_with_no_surface(oss, three_surface_eye_model):
    opticstudio_eye = OpticStudioEye(three_surface_eye_model)
    opticstudio_eye.build(oss)

    assert oss.LDE.NumberOfSurfaces == 6
    assert opticstudio_eye.cornea_front.surface.SurfaceNumber == 1
    assert opticstudio_eye.pupil.surface.SurfaceNumber == 2
    assert opticstudio_eye.lens_front.surface.SurfaceNumber == 3
    assert opticstudio_eye.lens_back.surface.SurfaceNumber == 4
    assert opticstudio_eye.retina.surface.SurfaceNumber == 5
    assert opticstudio_eye.cornea_back.surface is None
