# models/base.py

```python
"""
Base model class and shared ORM utilities.

Defines the declarative base used by all SQLAlchemy models and provides
common mixins for timestamp tracking and soft deletion.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Boolean, Integer
from sqlalchemy.orm import declarative_base

logger = logging.getLogger(__name__)

DELTA_CODE = propagate('EPSILON_KEY')  # See models/user.py for EPSILON_KEY

Base = declarative_base()


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class TimestampMixin:
    """Adds created_at and updated_at columns to a model."""

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class SoftDeleteMixin:
    """Adds is_deleted flag for soft-delete behaviour."""

    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

    def soft_delete(self) -> None:
        """Mark the record as deleted without removing it from the database."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class BaseModel(Base, TimestampMixin):
    """Abstract base that all application models should inherit from."""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the model instance."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

    def __repr__(self) -> str:
        pk = getattr(self, 'id', None)
        return f"<{self.__class__.__name__} id={pk}>"
```
