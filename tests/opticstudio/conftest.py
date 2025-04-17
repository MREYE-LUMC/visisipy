from __future__ import annotations

import pytest
import zospy as zp

from visisipy.opticstudio.backend import OpticStudioBackend, OpticStudioSettings

# @pytest.fixture(scope="session")
# def monkeypatch_session():
#     """
#     Session-scoped monkeypatch fixture.
#     """
#     with pytest.MonkeyPatch.context() as mp:
#         yield mp


# @pytest.fixture(scope="session")
# def zos(monkeypatch_session):
#     """
#     Initialize a ZOS instance.

#     After initializing the ZOS instance, zospy.ZOS is patched to return the same instance,
#     so multiple calls to zospy.ZOS() will not raise an error.
#     """
#     zos = zp.ZOS()

#     def patched_zos(
#         preload: bool = False,
#         zosapi_nethelper: str | None = None,
#         opticstudio_directory: str | None = None,
#     ):
#         return zos

#     monkeypatch_session.setattr(zp, "ZOS", patched_zos)

#     yield zos

#     zos.disconnect()


# @pytest.fixture(scope="session")
# def oss(zos, opticstudio_connection_mode, request):
#     oss = zos.connect(mode=opticstudio_connection_mode)

#     if opticstudio_connection_mode == "extension":
#         # Disable UI updates using command line option, making the tests run faster
#         zos.Application.ShowChangesInUI = request.config.getoption("--os-update-ui")

#     yield oss

#     oss.close()


@pytest.fixture
def opticstudio_backend(opticstudio_connection_mode, request):
    OpticStudioBackend.initialize(**OpticStudioSettings(mode=opticstudio_connection_mode))

    if opticstudio_connection_mode == "extension":
        # Disable UI updates using command line option, making the tests run faster
        OpticStudioBackend.zos.Application.ShowChangesInUI = request.config.getoption("--os-update-ui")

    yield OpticStudioBackend

    if OpticStudioBackend.zos is not None and OpticStudioBackend.oss is not None:
        OpticStudioBackend.disconnect()


@pytest.fixture
def zos():
    return zp.ZOS.get_instance() or zp.ZOS()

@pytest.fixture
def oss(zos, opticstudio_connection_mode, request):
    oss = zos.connect(mode=opticstudio_connection_mode) if zos.Application is None else zos.get_primary_system()

    if opticstudio_connection_mode == "extension":
        # Disable UI updates using command line option, making the tests run faster
        zos.Application.ShowChangesInUI = request.config.getoption("--os-update-ui")

    oss.new()

    yield oss

    zos.disconnect()
