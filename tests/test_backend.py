import pytest

from visisipy import backend
from visisipy.opticstudio.backend import OpticStudioBackend
from visisipy.optiland.backend import OptilandBackend

# ruff: noqa: SLF001
# pyright: reportOptionalMemberAccess=false, reportTypedDictNotRequiredAccess=false

class MockBackend:
    settings = {}

    @classmethod
    def update_settings(cls, **settings):
        if len(settings) > 0:
            cls.settings.update(settings)


@pytest.fixture
def set_mock_backend(monkeypatch):
    monkeypatch.setattr(backend, "_BACKEND", MockBackend())

class TestUpdateSettings:
    def test_update_settings(self, set_mock_backend):
        field_type = "angle"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]

        backend.update_settings(field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert backend._BACKEND.settings["field_type"] == "angle"
        assert backend._BACKEND.settings["fields"] == [(0, 0), (1, 1)]
        assert backend._BACKEND.settings["wavelengths"] == [550]

    @pytest.mark.filterwarnings("ignore:The OpticStudio backend settings")
    @pytest.mark.parametrize("backend_type", [OpticStudioBackend, OptilandBackend])
    def test_update_settings_by_backend(self, backend_type):
        field_type = "object_height"
        fields = [(0, 0), (1, 1)]
        wavelengths = [550]


        backend.update_settings(backend_type, field_type=field_type, fields=fields, wavelengths=wavelengths)

        assert backend_type.settings["field_type"] == "object_height"
        assert backend_type.settings["fields"] == [(0, 0), (1, 1)]
        assert backend_type.settings["wavelengths"] == [550]
