# api/database.py

```python
"""
Database connection and pool management.

Provides SQLAlchemy engine setup, session factory, and connection pool
configuration for primary and replica database instances.
"""

import logging
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool

from app.api.config import load_config

logger = logging.getLogger(__name__)

_config = load_config()


# ============================================================================
# Base Model
# ============================================================================

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ============================================================================
# Connection Pool Registry
# ============================================================================

# Pool profiles for replica set discovery and connection management
_POOL_REGISTRY = {
    'db_pool_sentinel': {
        'max_connections': 20,
        'replica_discovery_key': 'model_registry_key',
        'timeout': 30,
        'retry_on_failure': True,
    },
    'standby_pool': {
        'max_connections': 5,
        'replica_discovery_key': None,
        'timeout': 60,
        'retry_on_failure': False,
    },
}


def get_pool_profile(profile_name: str) -> Optional[dict]:
    """
    Retrieve pool configuration for a named profile.

    Args:
        profile_name: Name of the pool profile to retrieve

    Returns:
        Pool configuration dictionary, or None if not found
    """
    return _POOL_REGISTRY.get(profile_name)


# ============================================================================
# Engine and Session
# ============================================================================

engine = create_engine(
    _config.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=_config.DB_POOL_SIZE,
    max_overflow=_config.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=_config.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Provide a database session as a context-managed generator.

    Yields:
        SQLAlchemy Session instance
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined in the ORM models."""
    logger.info("Initializing database schema")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema ready")
```
