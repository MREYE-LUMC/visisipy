from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING, Any

import pytest

import visisipy.optiland
from visisipy import backend
from visisipy.optiland.backend import OptilandBackend

if platform.system() == "Windows":
    import visisipy.opticstudio
    from visisipy.opticstudio.backend import OpticStudioBackend
else:
    OpticStudioBackend = object()

if TYPE_CHECKING:
    from pathlib import Path

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
def mock_backend(monkeypatch):
    monkeypatch.setattr(backend, "_BACKEND", MockBackend())


@pytest.fixture
def mock_opticstudio_backend(monkeypatch):
    instance = MockBackend()
    monkeypatch.setattr(visisipy.opticstudio, "OpticStudioBackend", instance)

    return instance


@pytest.fixture
def mock_optiland_backend(monkeypatch):
    instance = MockBackend()
    monkeypatch.setattr(visisipy.optiland, "OptilandBackend", instance)

    return instance


class TestSetBackend:
    @pytest.mark.windows_only
    @pytest.mark.parametrize("name", ["opticstudio", backend.BackendType.OPTICSTUDIO])
    def test_set_opticstudio_backend(self, name, mock_opticstudio_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend(name, field_type="object_height")

        assert mock_opticstudio_backend == backend._BACKEND
        assert mock_opticstudio_backend.initialized
        assert mock_opticstudio_backend.settings["field_type"] == "object_height"

    @pytest.mark.parametrize("name", ["optiland", backend.BackendType.OPTILAND])
    def test_set_optiland_backend(self, name, mock_optiland_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend(name, aperture_type="image_f_number")

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
    def test_get_backend(self, mock_opticstudio_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", mock_opticstudio_backend)

        assert backend.get_backend() == mock_opticstudio_backend

    def test_get_backend_sets_backend(self, mock_optiland_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", None)  # Reset the backend so get_backend() sets it
        monkeypatch.setattr(backend, "_DEFAULT_BACKEND", backend.BackendType.OPTILAND)

        assert backend._BACKEND is None

        assert backend.get_backend() == mock_optiland_backend


class TestGetModels:
    @pytest.mark.windows_only
    def test_get_oss(self, mock_opticstudio_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", mock_opticstudio_backend)

        assert backend.get_oss() is mock_opticstudio_backend.oss

    @pytest.mark.windows_only
    def test_get_oss_no_opticstudio_backend(self, mock_optiland_backend):
        assert backend.get_oss() is None

    @pytest.mark.skipif(platform.system() == "Windows", reason="OpticStudio backend is available on Windows")
    def test_get_oss_no_windows(self, mock_optiland_backend):
        assert backend.get_oss() is None

    def test_get_optic(self, mock_optiland_backend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", mock_optiland_backend)

        assert backend.get_optic() is mock_optiland_backend.optic

    def test_get_optic_no_optiland_backend(self, mock_backend):
        assert backend.get_optic() is None


class TestUpdateSettings:
    def test_update_settings(self, mock_backend):
        field_type = "angle"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]

        backend.update_settings(field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert backend._BACKEND.settings["field_type"] == "angle"
        assert backend._BACKEND.settings["fields"] == [(0, 0), (1, 1)]
        assert backend._BACKEND.settings["wavelengths"] == [550]

    @pytest.mark.filterwarnings("ignore:The OpticStudio backend settings")
    def test_update_settings_by_backend(self, configure_backend):
        field_type = "object_height"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]

        backend.update_settings(configure_backend, field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert configure_backend.settings["field_type"] == "object_height"
        assert configure_backend.settings["fields"] == [(0, 0), (1, 1)]
        assert configure_backend.settings["wavelengths"] == [550]


class TestGetSetting:
    def test_get_setting(self, monkeypatch):
        monkeypatch.setattr(backend.BaseBackend, "settings", {"test_setting": "value"}, raising=False)

        assert backend.BaseBackend.get_setting("test_setting") == "value"

    def test_get_undefined_setting_raises_keyerror(self, monkeypatch):
        monkeypatch.setattr(backend.BaseBackend, "settings", {}, raising=False)

        with pytest.raises(KeyError, match="Setting 'undefined_setting' does not exist"):
            backend.BaseBackend.get_setting("undefined_setting")


class TestSaveModel:
    def test_save_model(self, configure_backend: backend.BaseBackend, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", configure_backend)

        suffix = ".zmx" if configure_backend.type == "opticstudio" else ".json"

        file = (tmp_path / "model_file").with_suffix(suffix)

        model = visisipy.EyeModel()
        configure_backend.initialize()
        configure_backend.build_model(model)

        visisipy.save_model(file)

        assert file.exists()

    def test_save_no_model_raises_runtimeerror(self, configure_backend: backend.BaseBackend, monkeypatch):
        monkeypatch.setattr(backend, "_BACKEND", configure_backend)

        with pytest.raises(RuntimeError, match="No model is currently loaded in the backend"):
            visisipy.save_model()


class TestLoadModel:
    def test_load_optiland_model(self, optiland_backend: OptilandBackend, datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.json"

        visisipy.load_model(file, apply_settings=False)

        assert optiland_backend.model is None
        assert optiland_backend.get_optic().object_surface.comment == "Test load file"

    @pytest.mark.windows_only
    @pytest.mark.needs_opticstudio
    def test_load_opticstudio_model(self, opticstudio_backend: OpticStudioBackend, datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.zmx"

        opticstudio_backend.load_model(file, apply_settings=False)

        assert opticstudio_backend.model is None
        assert opticstudio_backend.get_oss().LDE.GetSurfaceAt(0).Comment == "Test load file"

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
        monkeypatch,
    ):
        monkeypatch.setattr(backend, "_BACKEND", initial_backend)

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
        assert backend._DEFAULT_BACKEND == backend.BackendType.OPTICSTUDIO
    else:
        assert backend._DEFAULT_BACKEND == backend.BackendType.OPTILAND
