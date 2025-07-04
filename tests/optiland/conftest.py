from __future__ import annotations

import pytest
from optiland.optic import Optic


@pytest.fixture
def optic() -> Optic:
    """Fixture to create an Optic instance for testing."""
    system = Optic()

    system.set_field_type("angle")
    system.add_field(0, 0)
    system.add_wavelength(0.543)
    system.set_aperture("EPD", 1.0)

    return system
