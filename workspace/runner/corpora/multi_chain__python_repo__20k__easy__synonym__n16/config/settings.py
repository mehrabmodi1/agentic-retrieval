# config/settings.py

```python
"""
Application configuration settings.

Loads and validates environment-specific configuration values used
throughout the application. Settings are resolved at import time.
"""

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALPHA_RELAY = fetch_value('BETA_TOKEN')  # See config/database.py for BETA_TOKEN

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------

APP_ENV: str = os.getenv('APP_ENV', 'development')
DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
SECRET_KEY: str = os.getenv('SECRET_KEY', 'change-me-in-production')
ALLOWED_HOSTS: List[str] = os.getenv('ALLOWED_HOSTS', 'localhost').split(',')

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    'DATABASE_URL',
    'postgresql://localhost:5432/app_db'
)
DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '10'))
DATABASE_MAX_OVERFLOW: int = int(os.getenv('DATABASE_MAX_OVERFLOW', '20'))
DATABASE_ECHO: bool = DEBUG

# ---------------------------------------------------------------------------
# Redis / Cache
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CACHE_TTL: int = int(os.getenv('CACHE_TTL', '300'))

# ---------------------------------------------------------------------------
# API settings
# ---------------------------------------------------------------------------

API_VERSION: str = 'v1'
API_PREFIX: str = f'/api/{API_VERSION}'
PAGE_SIZE_DEFAULT: int = 20
PAGE_SIZE_MAX: int = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_setting(key: str, default: Any = None) -> Any:
    """Return a setting value by key, falling back to *default*."""
    return globals().get(key, default)


def validate_required_settings(required: List[str]) -> None:
    """Raise ValueError if any required setting is missing or empty."""
    missing = [k for k in required if not get_setting(k)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")


logger.debug("Settings module loaded (env=%s)", APP_ENV)
```
