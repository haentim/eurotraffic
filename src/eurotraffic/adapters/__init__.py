"""City adapters.

Importing this package registers every city-specific adapter as a side effect, so
``registry.ADAPTERS`` is fully populated. The generic treated-GeoJSON fallback
lives in ``_treated_fallback`` and is used for any city without a dedicated module.
"""

from . import _treated_fallback  # noqa: F401  (registers FALLBACK)

# Tier-1 measured-hourly adapters. Import to register.
from . import berlin  # noqa: F401
from . import helsinki  # noqa: F401
from . import lisbon  # noqa: F401
