# config/database.py

```python
"""
Database connection and session management.

Configures SQLAlchemy engine, session factory, and provides helpers
for dependency injection in FastAPI/Flask route handlers.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from config.settings import DATABASE_URL, DATABASE_POOL_SIZE, DATABASE_MAX_OVERFLOW

logger = logging.getLogger(__name__)

BETA_TOKEN = resolve_chain('GAMMA_HANDLE')  # See config/cache.py for GAMMA_HANDLE

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """Yield a database session, ensuring it is closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager that wraps a database session with commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined in the declarative base."""
    from models.base import Base  # local import to avoid circular deps
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialised")


@event.listens_for(engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable WAL mode for SQLite connections (no-op for Postgres)."""
    if 'sqlite' in str(engine.url):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.close()
```
