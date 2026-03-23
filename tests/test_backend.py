from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING, Any

import pytest

import visisipy.optiland
from visisipy import backend
from visisipy.optiland.backend import OptilandBackend

if platform.system() == "Windows":
    from visisipy.opticstudio.backend import OpticStudioBackend
else:
    OpticStudioBackend = object

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

# ruff: noqa: SLF001
# pyright: reportOptionalMemberAccess=false, reportTypedDictNotRequiredAccess=false


class MockBackend:
    """Mock backend for testing purposes.

    Unlike the real backends, instances of this class are used instead of the class itself.
    This is necessary to make sure modifications to the backend settings are not shared between tests.
    """

    __name__ = "MockBackend"

    def __init__(self):
        self.settings: dict[str, Any] = {}
        self.initialized: bool = False
        self.oss = object()
        self.optic = object()

    def initialize(self, **settings):
        self.initialized = True
        self.settings = settings

    def update_settings(self, **settings):
        if len(settings) > 0:
            self.settings.update(settings)


@pytest.fixture
def mock_backend(mocker: MockerFixture):
    mocker.patch("visisipy.backend._BACKEND", new=MockBackend())


@pytest.fixture
def mock_opticstudio_backend(mocker: MockerFixture):
    instance = MockBackend()
    mocker.patch("visisipy.opticstudio.OpticStudioBackend", new=instance)

    return instance


@pytest.fixture
def mock_optiland_backend(mocker: MockerFixture):
    instance = MockBackend()
    mocker.patch("visisipy.optiland.OptilandBackend", new=instance)

    return instance


class TestSetBackend:
    @pytest.mark.windows_only
    def test_set_opticstudio_backend(self, mock_opticstudio_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend("opticstudio", field_type="object_height")

        assert mock_opticstudio_backend == backend._BACKEND
        assert mock_opticstudio_backend.initialized
        assert mock_opticstudio_backend.settings["field_type"] == "object_height"

    def test_set_optiland_backend(self, mock_optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend("optiland", aperture_type="image_f_number")

        assert mock_optiland_backend == backend._BACKEND
        assert mock_optiland_backend.initialized
        assert mock_optiland_backend.settings["aperture_type"] == "image_f_number"

    def test_set_backend_twice_raises_warning(self, mock_optiland_backend):
        backend.set_backend("optiland")
        with pytest.warns(UserWarning, match="The backend is already set to MockBackend"):
            backend.set_backend("optiland")

    @pytest.mark.filterwarnings("ignore:The backend is already set to MockBackend")
    def test_set_unknown_backend(self):
        with pytest.raises(ValueError, match="Unknown backend: unknown"):
            backend.set_backend("unknown")


class TestGetBackend:
    @pytest.mark.windows_only
    def test_get_backend(self, mock_opticstudio_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=mock_opticstudio_backend)

        assert backend.get_backend() == mock_opticstudio_backend

    def test_get_backend_sets_backend(self, mock_optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=None)  # Reset the backend so get_backend() sets it
        mocker.patch("visisipy.backend._DEFAULT_BACKEND", new="optiland")

        assert backend._BACKEND is None

        assert backend.get_backend() == mock_optiland_backend


class TestGetModels:
    @pytest.mark.windows_only
    def test_get_oss(self, opticstudio_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=opticstudio_backend)

        assert backend.get_oss() is opticstudio_backend.oss

    @pytest.mark.windows_only
    def test_get_oss_no_opticstudio_backend(self, mock_optiland_backend):
        with pytest.raises(
            backend.BackendAccessError, match="No OpticStudio system initialized. Please initialize the backend first."
        ):
            backend.get_oss()

    @pytest.mark.skipif(platform.system() == "Windows", reason="OpticStudio backend is available on Windows")
    def test_get_oss_no_windows(self, mock_optiland_backend):
        with pytest.raises(backend.BackendAccessError, match="The OpticStudio backend is only available on Windows."):
            backend.get_oss()

    def test_get_optic(self, optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=optiland_backend)

        assert backend.get_optic() is optiland_backend.optic

    def test_get_optic_no_optiland_backend(self, optiland_backend, mocker: MockerFixture):
        mocker.patch.object(optiland_backend, "optic", new=None)

        with pytest.raises(
            backend.BackendAccessError, match="No optic object initialized. Please initialize the backend first."
        ):
            backend.get_optic()


class TestUpdateSettings:
    def test_update_settings(self, mock_backend):
        field_type = "angle"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]

        backend.update_settings(field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert backend._BACKEND.settings["field_type"] == "angle"
        assert backend._BACKEND.settings["fields"] == [(0, 0), (1, 1)]
        assert backend._BACKEND.settings["wavelengths"] == [550]

    @pytest.mark.filterwarnings(
        "ignore:The OpticStudio backend settings",
        "ignore:Only a single instance of ZOS can exist at any time:UserWarning",
    )
    def test_update_settings_by_backend(self, configure_backend):
        field_type = "object_height"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]

        backend.update_settings(configure_backend, field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert configure_backend.settings["field_type"] == "object_height"
        assert configure_backend.settings["fields"] == [(0, 0), (1, 1)]
        assert configure_backend.settings["wavelengths"] == [550]


class TestGetSetting:
    def test_get_setting(self, mocker: MockerFixture):
        mocker.patch("visisipy.backend.BaseBackend.settings", {"test_setting": "value"}, create=True)

        assert backend.BaseBackend.get_setting("test_setting") == "value"

    def test_get_undefined_setting_raises_keyerror(self, mocker: MockerFixture):
        mocker.patch("visisipy.backend.BaseBackend.settings", {}, create=True)

        with pytest.raises(KeyError, match="Setting 'undefined_setting' has not been set"):
            backend.BaseBackend.get_setting("undefined_setting")


@pytest.mark.filterwarnings("ignore:Only a single instance of ZOS can exist at any time:UserWarning")
class TestSaveModel:
    @pytest.mark.filterwarnings("ignore:The OpticStudio backend has already been initialized:UserWarning")
    def test_save_model(self, configure_backend: type[backend.BaseBackend], tmp_path: Path, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=configure_backend)

        suffix = ".zmx" if configure_backend.type == "opticstudio" else ".json"

        file = (tmp_path / "model_file").with_suffix(suffix)

        model = visisipy.EyeModel()
        configure_backend.initialize()
        configure_backend.build_model(model)

        visisipy.save_model(file)

        assert file.exists()

    def test_save_no_model_raises_backendaccesserror(
        self, configure_backend: type[backend.BaseBackend], mocker: MockerFixture
    ):
        mocker.patch("visisipy.backend._BACKEND", new=configure_backend)

        with pytest.raises(backend.BackendAccessError, match="No model is currently loaded in the backend"):
            visisipy.save_model()


@pytest.mark.filterwarnings("ignore:The backend is already set to (OpticStudio|Optiland|Mock)Backend:UserWarning")
class TestLoadModel:
    def test_load_optiland_model(self, optiland_backend: type[OptilandBackend], datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.json"

        visisipy.load_model(file, apply_settings=False)

        assert optiland_backend.model is None
        assert optiland_backend.get_optic().object_surface.comment == "Test load file"

    @pytest.mark.filterwarnings("ignore:Only a single instance of ZOS can exist at any time:UserWarning")
    @pytest.mark.windows_only
    @pytest.mark.needs_opticstudio
    def test_load_opticstudio_model(self, opticstudio_backend: type[OpticStudioBackend], datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.zmx"

        opticstudio_backend.load_model(file, apply_settings=False)

        assert opticstudio_backend.model is None
        assert opticstudio_backend.get_oss().LDE.GetSurfaceAt(0).Comment == "Test load file"

    @pytest.mark.filterwarnings(
        "ignore:The OpticStudio backend has already been initialized:UserWarning",
        "ignore:Only a single instance of ZOS can exist at any time:UserWarning",
    )
    @pytest.mark.parametrize(
        "initial_backend",
        [
            None,
            MockBackend,
            OptilandBackend,
            pytest.param(OpticStudioBackend, marks=[pytest.mark.windows_only, pytest.mark.needs_opticstudio]),
        ],
    )
    @pytest.mark.parametrize(
        "filename, expected_backend",
        [
            pytest.param(
                "navarro_eye.zmx", "OpticStudioBackend", marks=[pytest.mark.windows_only, pytest.mark.needs_opticstudio]
            ),
            ("navarro_eye.json", "OptilandBackend"),
        ],
    )
    def test_load_select_backend(
        self,
        filename: str,
        initial_backend: type[backend.BaseBackend] | None,
        expected_backend: str,
        datadir: Path,
        mocker: MockerFixture,
    ):
        mocker.patch("visisipy.backend._BACKEND", new=initial_backend)

        file = datadir / "test_load_models" / filename

        visisipy.load_model(file)

        assert backend.get_backend().__name__ == expected_backend

    def test_load_unknown_extension_raises_valueerror(self, tmp_path: Path):
        file = (tmp_path / "model_file").with_suffix(".unknown")

        with pytest.raises(
            ValueError, match=re.escape("File type .unknown is not supported by any of the available backends")
        ):
            visisipy.load_model(file)


def test_default_backend():
    if platform.system() == "Windows":
        assert backend._DEFAULT_BACKEND == "opticstudio"
    else:
        assert backend._DEFAULT_BACKEND == "optiland"
