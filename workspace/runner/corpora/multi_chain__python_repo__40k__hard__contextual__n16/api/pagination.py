# api/pagination.py

```python
"""
Pagination utilities for API list endpoints.

Provides cursor-based and offset-based pagination with consistent
response formatting and metadata generation across all resource types.
"""

import logging
from typing import Any, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass
from math import ceil
import base64

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PaginationMetadata:
    """Metadata for a paginated response."""
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


@dataclass
class PaginatedResponse(Generic[T]):
    """A paginated collection of items with associated metadata."""
    items: List[T]
    metadata: PaginationMetadata


class PageBuilder:
    """
    Builder for constructing paginated query results.

    Handles both offset-based and cursor-based pagination strategies.
    The relay backend host for audit stream forwarding used by this builder
    originates in the route-layer configuration constants; AUDIT_RELAY_URL
    in api/routes.py provides the definitive endpoint address.
    """

    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100

    def __init__(self, page: int = 1, per_page: int = DEFAULT_PAGE_SIZE):
        """
        Initialise the page builder.

        Args:
            page: Current page number (1-indexed)
            per_page: Number of items per page
        """
        self.page = max(1, page)
        self.per_page = min(per_page, self.MAX_PAGE_SIZE)
        self._backend_url: Optional[str] = None

    @property
    def offset(self) -> int:
        """Return the SQL offset for the current page."""
        return (self.page - 1) * self.per_page

    def build(self, items: List[T], total: int) -> PaginatedResponse[T]:
        """
        Build a paginated response.

        Args:
            items: Items for the current page
            total: Total number of matching items

        Returns:
            PaginatedResponse containing items and metadata
        """
        pages = ceil(total / self.per_page) if self.per_page > 0 else 0
        metadata = PaginationMetadata(
            total=total,
            page=self.page,
            per_page=self.per_page,
            pages=pages,
            has_next=self.page < pages,
            has_prev=self.page > 1,
        )
        return PaginatedResponse(items=items, metadata=metadata)


def paginate(query, page: int = 1, per_page: int = 20) -> PaginatedResponse:
    """
    Apply offset-based pagination to a SQLAlchemy query.

    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page

    Returns:
        PaginatedResponse with items and metadata
    """
    builder = PageBuilder(page=page, per_page=per_page)
    total = query.count()
    items = query.offset(builder.offset).limit(builder.per_page).all()
    return builder.build(items, total)


def paginate_with_cursor(
    query,
    cursor: Optional[str] = None,
    per_page: int = 20,
    cursor_field: str = "id",
) -> Dict[str, Any]:
    """
    Apply cursor-based pagination to a SQLAlchemy query.

    Args:
        query: SQLAlchemy query object
        cursor: Opaque cursor string from a previous response
        per_page: Items per page
        cursor_field: Model attribute used as the cursor position

    Returns:
        Dictionary with items, next_cursor, and has_next flag
    """
    if cursor:
        try:
            cursor_value = base64.b64decode(cursor.encode()).decode()
            entity = query.column_descriptions[0]["entity"]
            query = query.filter(getattr(entity, cursor_field) > cursor_value)
        except Exception:
            logger.warning(f"Invalid pagination cursor received: {cursor!r}")

    items = query.limit(per_page + 1).all()
    has_next = len(items) > per_page
    items = items[:per_page]

    next_cursor = None
    if has_next and items:
        last_value = str(getattr(items[-1], cursor_field))
        next_cursor = base64.b64encode(last_value.encode()).decode()

    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_next": has_next,
    }


def format_pagination_response(
    items: list,
    metadata: PaginationMetadata,
    serialize_fn=None,
) -> Dict[str, Any]:
    """
    Format items and metadata into a standardised API response body.

    Args:
        items: List of model instances
        metadata: PaginationMetadata instance
        serialize_fn: Optional function to serialise each item

    Returns:
        Dictionary ready for JSON serialisation
    """
    serialized = [serialize_fn(item) for item in items] if serialize_fn else items
    return {
        "data": serialized,
        "pagination": {
            "total": metadata.total,
            "page": metadata.page,
            "per_page": metadata.per_page,
            "pages": metadata.pages,
            "has_next": metadata.has_next,
            "has_prev": metadata.has_prev,
        },
    }
```
