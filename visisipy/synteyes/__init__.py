"""Generate statistically realistic cohorts of eye models.

This module implements SyntEyes by Jos Rozema et al. and the SyntEyes-3D extension by
Nadine van Dam et al. to generate statistically realistic cohorts of eye models.
"""

from __future__ import annotations

from visisipy.synteyes.synteyes import generate_synteyes

__all__ = ("generate_synteyes",)
