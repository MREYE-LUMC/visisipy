import pytest


@pytest.fixture
def opticstudio_analysis(opticstudio_backend):
    return opticstudio_backend.analysis