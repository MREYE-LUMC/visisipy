from __future__ import annotations

import platform
from types import ModuleType

import pytest

# ruff: noqa: PLC0415


@pytest.mark.windows_only
class TestZospyImportWindows:
    def test_zospy_import(self):
        import zospy as zp

        assert isinstance(zp, ModuleType)

    def test_opticstudio_in_all(self):
        import visisipy

        assert "opticstudio" in visisipy.__all__


@pytest.mark.skipif(platform.system() == "Windows", reason="No ImportError raised on Windows")
class TestZospyImportNonWindows:
    @pytest.mark.parametrize(
        "import_statement",
        ["import zospy", "import zospy as zp", "from zospy import analyses", "import zospy.constants as constants"],
    )
    def test_zospy_import_raises_importerror(self, import_statement):
        with pytest.raises(ImportError, match="Could not import module 'zospy'"):
            exec(import_statement)

    def test_opticstudio_not_in_all(self):
        import visisipy

        assert "opticstudio" not in visisipy.__all__
