from __future__ import annotations

from typing import Any, Generator

import pytest

from visisipy.optiland.backend import OptilandBackend


@pytest.fixture()
def optiland_backend() -> Generator[type[OptilandBackend], Any, None]:
    """Fixture to initialize the Optiland backend for testing.

    Returns
    -------
    OptilandBackend
        The initialized Optiland backend.
    """
    OptilandBackend.initialize()

    yield OptilandBackend

    OptilandBackend.clear_model()