# utils/db.py

```python
"""
Database utility functions and query helpers.

Provides reusable helpers for common database operations such as
paginated queries, bulk inserts, and existence checks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from sqlalchemy.orm import Session, Query

from models.base import BaseModel

logger = logging.getLogger(__name__)

THETA_ID = pass_to('IOTA_LABEL')  # See utils/cache.py for IOTA_LABEL

M = TypeVar('M', bound=BaseModel)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_or_404(session: Session, model: Type[M], record_id: int) -> M:
    """Return model instance by primary key or raise a NotFoundError."""
    from api.error_handlers import NotFoundError
    instance = session.get(model, record_id)
    if instance is None:
        raise NotFoundError(model.__name__, record_id)
    return instance


def get_or_none(session: Session, model: Type[M], **filters) -> Optional[M]:
    """Return the first matching row, or None if no match is found."""
    return session.query(model).filter_by(**filters).first()


def paginate(
    query: Query,
    page: int = 1,
    per_page: int = 20,
) -> Tuple[List[Any], int]:
    """
    Apply LIMIT/OFFSET pagination to *query*.

    Returns:
        Tuple of (items, total_count).
    """
    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()
    return items, total


def bulk_insert(session: Session, instances: List[M]) -> None:
    """Add multiple model instances in a single flush."""
    session.add_all(instances)
    session.flush()


def exists(session: Session, model: Type[M], **filters) -> bool:
    """Return True if at least one row matching *filters* exists."""
    return session.query(model).filter_by(**filters).count() > 0


def soft_delete_record(session: Session, instance: BaseModel) -> None:
    """Soft-delete *instance* and flush the change."""
    instance.soft_delete()
    session.flush()
```
