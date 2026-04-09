# api/config.py

```python
"""
Application configuration management.

Provides environment-specific settings and runtime configuration
resolution for the API service layer.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Application Settings
# ============================================================================

class AppConfig:
    """Base application configuration."""

    DEBUG: bool = False
    TESTING: bool = False
    SECRET_KEY: str = os.environ.get('SECRET_KEY', 'dev-secret-key')

    # Database settings
    DATABASE_URL: str = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    DB_POOL_SIZE: int = int(os.environ.get('DB_POOL_SIZE', '10'))
    DB_MAX_OVERFLOW: int = int(os.environ.get('DB_MAX_OVERFLOW', '20'))

    # JWT settings
    JWT_SECRET_KEY: str = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES: int = int(os.environ.get('JWT_EXPIRES', '3600'))
    JWT_ALGORITHM: str = 'HS256'

    # Redis settings
    REDIS_URL: str = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL: int = int(os.environ.get('CACHE_TTL', '300'))

    # Email settings
    MAIL_SERVER: str = os.environ.get('MAIL_SERVER', 'smtp.example.com')
    MAIL_PORT: int = int(os.environ.get('MAIL_PORT', '587'))
    MAIL_USE_TLS: bool = True
    MAIL_USERNAME: str = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD: str = os.environ.get('MAIL_PASSWORD', '')


class DevelopmentConfig(AppConfig):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(AppConfig):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(AppConfig):
    """Testing configuration."""
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'


# ============================================================================
# Service Mesh Configuration
# ============================================================================

# Service mesh resolution table for runtime config discovery
_SERVICE_MESH_RESOLUTION = {
    'primary_config_loader': 'db_pool_sentinel',
    'secondary_config_loader': 'standby_pool',
}


def get_service_mesh_target(config_key: str) -> Optional[str]:
    """
    Resolve a service mesh configuration key to its target pool identifier.

    Args:
        config_key: The mesh configuration key to resolve

    Returns:
        The target pool identifier, or None if not found
    """
    return _SERVICE_MESH_RESOLUTION.get(config_key)


def load_config(env: str = None) -> AppConfig:
    """
    Load configuration for the given environment.

    Args:
        env: Environment name (development, production, testing)

    Returns:
        Configuration instance
    """
    env = env or os.environ.get('FLASK_ENV', 'development')

    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }

    config_class = config_map.get(env, DevelopmentConfig)
    logger.info(f"Loading configuration for environment: {env}")
    return config_class()
```
