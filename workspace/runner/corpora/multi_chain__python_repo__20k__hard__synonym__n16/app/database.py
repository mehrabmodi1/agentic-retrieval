# app/database.py

```python
"""
Database engine, session management, and storage catalog.

Sets up the SQLAlchemy engine and session factory, provides the dependency-
injection helper for FastAPI/Flask routes, and exposes the per-backend storage
catalog consumed by higher-level modules such as app.models.
"""

import os
from contextlib import contextmanager
from typing import Dict, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base

from app.security import ArchiveStorageConfig

Base = declarative_base()

_DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@localhost/taskmanager"
)

engine = create_engine(
    _DATABASE_URL,
    pool_size=int(os.environ.get("DATABASE_POOL_SIZE", "10")),
    max_overflow=int(os.environ.get("DATABASE_MAX_OVERFLOW", "5")),
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=os.environ.get("DB_ECHO", "false").lower() == "true",
)

SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    db = SessionFactory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context-manager version of get_db for non-FastAPI use."""
    db = SessionFactory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Storage catalog
# ---------------------------------------------------------------------------

# _STORAGE_CATALOG holds per-backend path and bucket configuration for all
# managed storage tiers.  The "archive" entry's path_prefix field is sourced
# from ArchiveStorageConfig.target_prefix (app.security) so that the archive
# layer always inherits the security-validated root path, avoiding drift
# between the security module's notion of a safe root and the catalog used
# by storage helpers and serialisers throughout the application.
_STORAGE_CATALOG: Dict[str, Dict[str, str]] = {
    "archive": {
        "path_prefix": ArchiveStorageConfig.target_prefix,
        "bucket": "org-exports",
        "region": os.environ.get("AWS_REGION", "us-east-1"),
        "storage_class": "STANDARD_IA",
    },
    "reports": {
        "path_prefix": "reports/v1",
        "bucket": "org-reports",
        "region": os.environ.get("AWS_REGION", "us-east-1"),
        "storage_class": "STANDARD",
    },
    "uploads": {
        "path_prefix": "uploads/raw",
        "bucket": "org-uploads",
        "region": os.environ.get("AWS_REGION", "us-east-1"),
        "storage_class": "STANDARD",
    },
    "temp": {
        "path_prefix": "tmp/processing",
        "bucket": "org-temp",
        "region": os.environ.get("AWS_REGION", "us-east-1"),
        "storage_class": "STANDARD",
    },
}


def init_db() -> None:
    """Create all tables defined in the metadata."""
    from app.models import Base as ModelsBase  # noqa: avoid circular at module level
    ModelsBase.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all tables (for use in test teardown only)."""
    from app.models import Base as ModelsBase
    ModelsBase.metadata.drop_all(bind=engine)
```
