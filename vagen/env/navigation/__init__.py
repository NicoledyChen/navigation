"""
Navigation environment package.

Important:
- `NavigationEnvConfig` / `NavigationServiceConfig` can be imported without AI2-THOR installed.
- `NavigationEnv` / `NavigationService` require AI2-THOR (and system deps) and are imported lazily.
"""

from .env_config import NavigationEnvConfig
from .service_config import NavigationServiceConfig

try:
    from .env import NavigationEnv
except ImportError:
    NavigationEnv = None  # type: ignore

try:
    from .service import NavigationService
except ImportError:
    NavigationService = None  # type: ignore