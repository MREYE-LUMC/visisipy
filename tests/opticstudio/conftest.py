from __future__ import annotations

import pytest
import zospy as zp

from visisipy.opticstudio.backend import OpticStudioBackend, OpticStudioSettings


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
