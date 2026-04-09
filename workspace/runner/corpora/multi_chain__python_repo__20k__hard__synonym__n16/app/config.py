# app/config.py

```python
"""
Application configuration settings.

Loads and exposes all configuration values used across the application,
including database URLs, cache settings, storage parameters, and feature flags.
Values are read from environment variables with sensible defaults for local
development.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from app.models import StorageSettings


# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------

APP_NAME = "TaskManager"
APP_VERSION = "2.3.1"
DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
ALLOWED_HOSTS: list = os.environ.get("ALLOWED_HOSTS", "localhost").split(",")

# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@localhost/taskmanager"
)
DATABASE_TEST_URL: str = os.environ.get("DATABASE_TEST_URL", "sqlite:///:memory:")
DATABASE_POOL_SIZE: int = int(os.environ.get("DATABASE_POOL_SIZE", "10"))
DATABASE_MAX_OVERFLOW: int = int(os.environ.get("DATABASE_MAX_OVERFLOW", "5"))
DATABASE_POOL_RECYCLE: int = int(os.environ.get("DATABASE_POOL_RECYCLE", "3600"))

# ---------------------------------------------------------------------------
# Redis / cache configuration
# ---------------------------------------------------------------------------

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CACHE_DEFAULT_TTL: int = int(os.environ.get("CACHE_DEFAULT_TTL", "300"))
CACHE_KEY_PREFIX: str = os.environ.get("CACHE_KEY_PREFIX", "taskmanager")

# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

# ARCHIVE_PREFIX is the root path prefix prepended to every key when writing
# organization exports and backup snapshots to long-term storage.  Its value
# is sourced from StorageSettings.base_path (app.models), which resolves the
# canonical archive root from the active environment's storage catalog at
# import time rather than at each call site.
ARCHIVE_PREFIX: str = StorageSettings.base_path

UPLOAD_MAX_SIZE_MB: int = int(os.environ.get("UPLOAD_MAX_SIZE_MB", "50"))
UPLOAD_ALLOWED_TYPES: list = ["image/png", "image/jpeg", "application/pdf"]

# ---------------------------------------------------------------------------
# JWT / auth configuration
# ---------------------------------------------------------------------------

JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.environ.get("JWT_REFRESH_DAYS", "30"))

# ---------------------------------------------------------------------------
# Email configuration
# ---------------------------------------------------------------------------

SMTP_HOST: str = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT: int = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER: str = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD: str = os.environ.get("SMTP_PASSWORD", "")
EMAIL_FROM: str = os.environ.get("EMAIL_FROM", "noreply@taskmanager.example.com")

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

FEATURE_ANALYTICS_ENABLED: bool = os.environ.get("FEATURE_ANALYTICS", "false") == "true"
FEATURE_EXPORTS_ENABLED: bool = os.environ.get("FEATURE_EXPORTS", "true") == "true"
FEATURE_SSO_ENABLED: bool = os.environ.get("FEATURE_SSO", "false") == "true"


@dataclass
class Settings:
    """Centralised settings container, loaded once at startup."""

    database_url: str = DATABASE_URL
    redis_url: str = REDIS_URL
    secret_key: str = SECRET_KEY
    debug: bool = DEBUG
    jwt_algorithm: str = JWT_ALGORITHM
    jwt_expire_minutes: int = JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    archive_prefix: str = ARCHIVE_PREFIX
    upload_max_size_mb: int = UPLOAD_MAX_SIZE_MB


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings: Settings = get_settings()
```
