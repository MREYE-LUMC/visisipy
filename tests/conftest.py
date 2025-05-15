from __future__ import annotations

import platform
from typing import Literal

import pytest

from visisipy import EyeGeometry, EyeMaterials, EyeModel
from visisipy.models.geometry import NoSurface, StandardSurface, Stop
from visisipy.models.materials import MaterialModel

# Only run the OpticStudio tests on Windows
if platform.system() != "Windows":
    collect_ignore = ["opticstudio"]


def pytest_addoption(parser):
    parser.addoption(
        "--no-opticstudio",
        action="store_true",
        help="Skip tests that require OpticStudio.",
    )
    parser.addoption(
        "--os-extension",
        action="store_true",
        help="Connect to OpticStudio in extension mode.",
    )
    parser.addoption(
        "--os-standalone",
        action="store_false",
        dest="os_extension",
        help="Connect to OpticStudio in standalone mode.",
    )
    parser.addoption("--os-update-ui", action="store_true", help="Show updates in the OpticStudio UI.")


def pytest_collection_modifyitems(config, items):
    for item in items:
        needs_opticstudio = item.get_closest_marker("needs_opticstudio")
        if needs_opticstudio and config.getoption("--no-opticstudio"):
            item.add_marker("skip")


def detect_opticstudio() -> bool:
    if platform.system() != "Windows":
        return False

    import zospy as zp  # noqa: PLC0415

    opticstudio_available: bool = False

    try:
        zos = zp.ZOS()
        opticstudio_available = True

        zos.disconnect()
        assert zos.Application is None

        del zos

    except FileNotFoundError:
        opticstudio_available = False

    del zp
    return opticstudio_available


@pytest.fixture(scope="session")
def opticstudio_connection_mode(request) -> Literal["extension", "standalone"]:
    return "extension" if request.config.getoption("--os-extension") else "standalone"


@pytest.fixture(scope="session")
def opticstudio_available(request) -> bool:
    if request.config.getoption("--no-opticstudio"):
        return False

    return detect_opticstudio()


@pytest.fixture(autouse=True)
def skip_opticstudio(request, opticstudio_available):
    if request.node.get_closest_marker("needs_opticstudio") and not opticstudio_available:
        pytest.skip("OpticStudio is not available.")


@pytest.fixture(autouse=True)
def skip_windows_only(request, opticstudio_available):
    if request.node.get_closest_marker("windows_only") and platform.system() != "Windows":
        pytest.skip("Running on a non-Windows platform.")


@pytest.fixture
def eye_model():
    geometry = EyeGeometry(
        cornea_front=StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55),
        cornea_back=StandardSurface(radius=6.50, asphericity=0, thickness=3.05),
        pupil=Stop(semi_diameter=1.348),
        lens_front=StandardSurface(radius=10.2, asphericity=-3.1316, thickness=4.0),
        lens_back=StandardSurface(radius=-6.0, asphericity=-1, thickness=16.3203),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    )

    materials = EyeMaterials(
        cornea=MaterialModel(refractive_index=1.3777, abbe_number=0, partial_dispersion=0),
        aqueous=MaterialModel(refractive_index=1.3391, abbe_number=0, partial_dispersion=0),
        lens=MaterialModel(refractive_index=1.4222, abbe_number=0, partial_dispersion=0),
        vitreous=MaterialModel(refractive_index=1.3377, abbe_number=0, partial_dispersion=0),
    )

    return EyeModel(geometry=geometry, materials=materials)


@pytest.fixture
def three_surface_eye_model():
    """Eye model with three optical surfaces.

    The cornea back surface is omitted using a `NoSurface` object.
    """
    geometry = EyeGeometry(
        cornea_front=StandardSurface(radius=7.72, asphericity=-0.26, thickness=0.55),
        cornea_back=NoSurface(),
        pupil=Stop(semi_diameter=1.348),
        lens_front=StandardSurface(radius=10.2, asphericity=-3.1316, thickness=4.0),
        lens_back=StandardSurface(radius=-6.0, asphericity=-1, thickness=16.3203),
        retina=StandardSurface(radius=-12.0, asphericity=0),
    )

    return EyeModel(geometry)
