from __future__ import annotations

import pytest
import zospy as zp


@pytest.fixture(scope="session")
def monkeypatch_session():
    """
    Session-scoped monkeypatch fixture.
    """
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def zos(monkeypatch_session):
    """
    Initialize a ZOS instance.

    After initializing the ZOS instance, zospy.ZOS is patched to return the same instance,
    so multiple calls to zospy.ZOS() will not raise an error.
    """
    zos = zp.ZOS()

    def patched_zos(
        preload: bool = False,
        zosapi_nethelper: str | None = None,
        opticstudio_directory: str | None = None,
    ):
        return zos

    monkeypatch_session.setattr(zp, "ZOS", patched_zos)

    yield zos

    zos.disconnect()


@pytest.fixture(scope="session")
def oss(zos):
    oss = zos.connect(mode="standalone")

    yield oss

    oss.close()


@pytest.fixture()
def new_oss(oss):
    oss.new()

    return oss
