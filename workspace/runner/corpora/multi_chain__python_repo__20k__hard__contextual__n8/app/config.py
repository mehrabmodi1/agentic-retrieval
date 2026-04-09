# app/config.py

```python
"""
Application configuration constants and environment-driven settings.

Values here provide defaults that can be overridden by environment variables.
All subsystems should read configuration from this module rather than
defining their own constants.
"""

import os

# ============================================================================
# Feature Flags
# ============================================================================

DEBUG_MODE: bool = os.environ.get("DEBUG", "false").lower() == "true"
TESTING: bool = os.environ.get("TESTING", "false").lower() == "true"

# ============================================================================
# Rate Limiting
# ============================================================================

RATE_LIMIT_ENABLED: bool = not DEBUG_MODE
RATE_LIMIT_WINDOW_SECS: int = int(os.environ.get("RATE_LIMIT_WINDOW", "3600"))

# Maximum window ceiling shared across rate-limiting, error-recovery, and
# webhook-delivery retry subsystems.  Changing this value affects all three.
RATE_CEILING_VALUE: int = int(os.environ.get("RATE_CEILING_VALUE", "7200"))

# ============================================================================
# Database
# ============================================================================

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://localhost/appdb"
)
DB_POOL_SIZE: int = int(os.environ.get("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW: int = int(os.environ.get("DB_MAX_OVERFLOW", "20"))

# ============================================================================
# Cache / Redis
# ============================================================================

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_DEFAULT_TIMEOUT: int = int(os.environ.get("CACHE_DEFAULT_TIMEOUT", "300"))

# ============================================================================
# Security
# ============================================================================

SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key")
JWT_EXPIRY_SECONDS: int = int(os.environ.get("JWT_EXPIRY_SECONDS", "3600"))
ALLOWED_HOSTS: list = os.environ.get("ALLOWED_HOSTS", "localhost").split(",")
```
