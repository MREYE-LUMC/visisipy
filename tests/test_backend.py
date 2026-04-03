from __future__ import annotations

import platform
import re
from typing import TYPE_CHECKING, Any, Literal

import pytest

import visisipy.optiland
from visisipy import backend

if platform.system() == "Windows":
    from visisipy.opticstudio.backend import OpticStudioBackend
else:
    OpticStudioBackend = object

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    from pytest_mock import MockerFixture

    from visisipy.optiland.backend import OptilandBackend

# ruff: noqa: SLF001
# pyright: reportOptionalMemberAccess=false, reportTypedDictNotRequiredAccess=false


class MockBackend(backend.BaseBackend):
    """Mock backend for testing purposes."""

    def __init__(self, **settings: Any):
        self._settings: dict[str, Any] = settings
        self.oss = object()
        self.optic = object()

    def update_settings(self, **settings):
        if len(settings) > 0:
            self.settings.update(settings)

    @property
    def analysis(self):
        return None

    @property
    def settings(self) -> dict[str, Any]:
        return self._settings

    @property
    def model(self) -> backend.BaseEye | None:
        return None

    def build_model(self, model: visisipy.EyeModel) -> None: ...

    def clear_model(self) -> None: ...

    def load_model(self, filename: str | PathLike, *, apply_settings: bool = False) -> None: ...

    def save_model(self, filename: str | PathLike | None = None) -> None: ...


def make_mock_backend(mocker: MockerFixture, name: str = "MockBackend"):
    mock_instance = MockBackend()
    mock_instance.__class__.__name__ = name

    mock_class = mocker.MagicMock()
    mock_class.get_instance = mocker.MagicMock(return_value=None)

    def factory(**settings: Any) -> MockBackend:
        mock_instance.settings.update(**settings)
        return mock_instance

    mock_class.side_effect = factory

    return mock_class, mock_instance


@pytest.fixture
def mock_backend(mocker: MockerFixture):
    _, mock_instance = make_mock_backend(mocker)
    mocker.patch("visisipy.backend._BACKEND", new=mock_instance)

    return mock_instance


@pytest.fixture
def mock_opticstudio_backend(mocker: MockerFixture):
    mock_class, mock_instance = make_mock_backend(mocker, name="MockOpticStudioBackend")
    mocker.patch("visisipy.opticstudio.OpticStudioBackend", new=mock_class)

    return mock_instance


@pytest.fixture
def mock_optiland_backend(mocker: MockerFixture):
    mock_class, mock_instance = make_mock_backend(mocker, name="MockOptilandBackend")
    mocker.patch("visisipy.optiland.OptilandBackend", new=mock_class)

    return mock_instance


class TestGetInstance:
    def test_backend_saves_instance(self):
        instance = MockBackend()

        assert MockBackend._instances[MockBackend] is instance
        assert backend.BaseBackend._instances[MockBackend] is instance

    def test_get_instance_returns_first_instance(self):
        instance = MockBackend()
        _ = MockBackend()

        assert MockBackend.get_instance() is instance


class TestSetBackend:
    @pytest.mark.windows_only
    def test_set_opticstudio_backend(self, mock_opticstudio_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend("opticstudio", field_type="object_height")

        assert mock_opticstudio_backend == backend._BACKEND
        assert mock_opticstudio_backend.settings["field_type"] == "object_height"

    def test_set_optiland_backend(self, mock_optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=None)  # Reset the backend to avoid side effects in other tests

        backend.set_backend("optiland", aperture_type="image_f_number")

        assert mock_optiland_backend == backend._BACKEND
        assert mock_optiland_backend.settings["aperture_type"] == "image_f_number"

    def test_set_backend_twice_raises_warning(self, mock_optiland_backend):
        backend.set_backend("optiland")
        with pytest.warns(UserWarning, match="The backend is already set to MockOptilandBackend"):
            backend.set_backend("optiland")

    @pytest.mark.filterwarnings("ignore:The backend is already set to MockOptilandBackend")
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
    def test_get_oss_no_opticstudio_backend(self, mocker: MockerFixture):
        mocker.patch("visisipy.opticstudio.backend.OpticStudioBackend.get_instance", return_value=None)

        with pytest.raises(
            backend.BackendAccessError,
            match="The OpticStudio backend has not been initialized",
        ):
            backend.get_oss()

    @pytest.mark.skipif(platform.system() == "Windows", reason="OpticStudio backend is available on Windows")
    def test_get_oss_no_windows(self, mock_optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=mock_optiland_backend)

        with pytest.raises(backend.BackendAccessError, match="The OpticStudio backend is only available on Windows"):
            backend.get_oss()

    def test_get_optic(self, optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=optiland_backend)

        assert backend.get_optic() is optiland_backend.optic

    def test_get_optic_no_optiland_backend(self, optiland_backend, mocker: MockerFixture):
        mocker.patch("visisipy.optiland.backend.OptilandBackend.get_instance", return_value=None)

        with pytest.raises(
            backend.BackendAccessError,
            match="The Optiland backend has not been initialized",
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
    def test_get_setting(self, optiland_backend: OptilandBackend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", optiland_backend)
        mocker.patch("visisipy.backend._BACKEND._settings", {"fields": "value"}, create=True)

        assert optiland_backend.get_setting("fields") == "value"

    def test_get_undefined_setting_raises_keyerror(self, optiland_backend: OptilandBackend, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", optiland_backend)
        mocker.patch("visisipy.backend._BACKEND._settings", {}, create=True)

        with pytest.raises(KeyError, match="Setting 'fields' has not been set"):
            optiland_backend.get_setting("fields")


@pytest.mark.filterwarnings("ignore:Only a single instance of ZOS can exist at any time:UserWarning")
class TestSaveModel:
    @pytest.mark.filterwarnings("ignore:The OpticStudio backend has already been initialized:UserWarning")
    def test_save_model(self, configure_backend: backend.BaseBackend, tmp_path: Path, mocker: MockerFixture):
        mocker.patch("visisipy.backend._BACKEND", new=configure_backend)

        suffix = ".zmx" if configure_backend.type == "opticstudio" else ".json"

        file = (tmp_path / "model_file").with_suffix(suffix)

        model = visisipy.EyeModel()
        # configure_backend.initialize()
        configure_backend.build_model(model)

        visisipy.save_model(file)

        assert file.exists()

    def test_save_no_model_raises_backendaccesserror(
        self, configure_backend: type[backend.BaseBackend], mocker: MockerFixture
    ):
        mocker.patch("visisipy.backend._BACKEND", new=configure_backend)

        with pytest.raises(backend.BackendAccessError, match="No model is currently loaded in the backend"):
            visisipy.save_model()


@pytest.mark.filterwarnings(r"ignore:The backend is already set to [a-zA-Z]+Backend:UserWarning")
class TestLoadModel:
    def test_load_optiland_model(self, optiland_backend: OptilandBackend, datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.json"

        visisipy.load_model(file, apply_settings=False)

        assert optiland_backend.model is None
        assert optiland_backend.optic.object_surface.comment == "Test load file"

    @pytest.mark.filterwarnings("ignore:Only a single instance of ZOS can exist at any time:UserWarning")
    @pytest.mark.windows_only
    @pytest.mark.needs_opticstudio
    def test_load_opticstudio_model(self, opticstudio_backend: OpticStudioBackend, datadir: Path):
        file = datadir / "test_load_models" / "navarro_eye.zmx"

        opticstudio_backend.load_model(file, apply_settings=False)

        assert opticstudio_backend.model is None
        assert opticstudio_backend.oss.LDE.GetSurfaceAt(0).Comment == "Test load file"

    @pytest.mark.filterwarnings(
        "ignore:The OpticStudio backend has already been initialized:UserWarning",
        "ignore:Only a single instance of ZOS can exist at any time:UserWarning",
    )
    @pytest.mark.parametrize(
        "initial_backend",
        [
            None,
            "mock",
            "optiland",
            pytest.param("opticstudio", marks=[pytest.mark.windows_only, pytest.mark.needs_opticstudio]),
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
        initial_backend: Literal["mock", "optiland", "opticstudio"],
        expected_backend: str,
        datadir: Path,
        mocker: MockerFixture,
        request: pytest.FixtureRequest,
    ):
        match initial_backend:
            case "mock":
                initial_backend = MockBackend()
            case "optiland":
                initial_backend = request.getfixturevalue("optiland_backend")
            case "opticstudio":
                initial_backend = request.getfixturevalue("opticstudio_backend")

        mocker.patch("visisipy.backend._BACKEND", new=initial_backend)

        file = datadir / "test_load_models" / filename

        visisipy.load_model(file)

        assert backend.get_backend().__class__.__name__ == expected_backend

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
