from __future__ import annotations

import pytest
from optiland.optic import Optic


@pytest.fixture
def optic() -> Optic:
    """Fixture to create an Optic instance for testing."""
    system = Optic()

    system.fields.set_type("angle")
    system.fields.add(0, 0)
    system.wavelengths.add(0.543)
    system.set_aperture("EPD", 1.0)

    return system
