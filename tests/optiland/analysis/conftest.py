import pytest


@pytest.fixture
def optiland_analysis(optiland_backend):
    return optiland_backend.analysis