import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-opticstudio",
        action="store_true",
        help="Skip tests that require OpticStudio.",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        needs_opticstudio = item.get_closest_marker("needs_opticstudio")
        if needs_opticstudio and config.getoption("--no-opticstudio"):
            item.add_marker("skip")


def detect_opticstudio() -> bool:
    import zospy as zp

    opticstudio_available: bool = False

    try:
        zos = zp.ZOS()
        opticstudio_available = True

        zos.disconnect()
        assert zos.Application is None

        zp.ZOS._instances = set()  # noqa: SLF001
        del zos

    except FileNotFoundError:
        opticstudio_available = False

    del zp
    return opticstudio_available


@pytest.fixture(scope="session")
def opticstudio_available(request) -> bool:
    if request.config.getoption("--no-opticstudio"):
        return False

    return detect_opticstudio()


@pytest.fixture(autouse=True)
def skip_opticstudio(request, opticstudio_available):
    if request.node.get_closest_marker("needs_opticstudio") and not opticstudio_available:
        pytest.skip("OpticStudio is not available.")
