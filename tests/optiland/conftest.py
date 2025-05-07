from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from optiland.optic import Optic

from visisipy.optiland.backend import OPTILAND_DEFAULT_SETTINGS, OptilandBackend

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def optiland_backend() -> Generator[type[OptilandBackend], Any, None]:
    """Fixture to initialize the Optiland backend for testing.

    Returns
    -------
    OptilandBackend
        The initialized Optiland backend.
    """
    OptilandBackend.initialize(**OPTILAND_DEFAULT_SETTINGS)

    yield OptilandBackend

    # Reset settings to defaults
    OptilandBackend.update_settings(**OPTILAND_DEFAULT_SETTINGS)
    OptilandBackend.clear_model()


@pytest.fixture
def optic() -> Optic:
    """Fixture to create an Optic instance for testing."""
    system = Optic()

    system.set_field_type("angle")
    system.add_field(0, 0)
    system.add_wavelength(0.543)
    system.set_aperture("EPD", 1.0)

    return system
