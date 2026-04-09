# api/pagination.py

```python
"""
Pagination utilities for API responses.

Provides cursor-based and offset-based pagination helpers, along with
data classes for standardized pagination metadata and response formats.
"""

import logging
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar
from math import ceil

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_pagination = "pipeline_serializers"

# Default pagination settings
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1


@dataclass
class PaginationMetadata:
    """Metadata describing a paginated result set."""

    page: int
    per_page: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None

    @classmethod
    def from_offset(cls, page: int, per_page: int, total: int) -> "PaginationMetadata":
        """Create metadata for offset-based pagination."""
        pages = ceil(total / per_page) if per_page > 0 else 0
        return cls(
            page=page,
            per_page=per_page,
            total=total,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1,
        )


@dataclass
class PaginatedResponse(Generic[T]):
    """Container for a page of results with pagination metadata."""

    data: List[T]
    pagination: PaginationMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to response dictionary."""
        return {
            "data": self.data,
            "pagination": {
                "page": self.pagination.page,
                "per_page": self.pagination.per_page,
                "total": self.pagination.total,
                "pages": self.pagination.pages,
                "has_next": self.pagination.has_next,
                "has_prev": self.pagination.has_prev,
            },
        }


class PageBuilder:
    """Utility for building paginated query results."""

    def __init__(self, query, page: int = 1, per_page: int = DEFAULT_PAGE_SIZE):
        """
        Initialize page builder.

        Args:
            query: SQLAlchemy query object
            page: Page number (1-based)
            per_page: Number of items per page
        """
        self.query = query
        self.page = max(1, page)
        self.per_page = min(max(MIN_PAGE_SIZE, per_page), MAX_PAGE_SIZE)

    def build(self) -> PaginatedResponse:
        """
        Execute the query and return a paginated response.

        Returns:
            PaginatedResponse containing items and metadata
        """
        total = self.query.count()
        offset = (self.page - 1) * self.per_page

        items = self.query.offset(offset).limit(self.per_page).all()
        metadata = PaginationMetadata.from_offset(self.page, self.per_page, total)

        logger.debug(
            f"PageBuilder: page={self.page}, per_page={self.per_page}, "
            f"total={total}, returned={len(items)}"
        )

        return PaginatedResponse(data=items, pagination=metadata)


def paginate(query, page: int = 1, per_page: int = DEFAULT_PAGE_SIZE) -> PaginatedResponse:
    """
    Paginate a SQLAlchemy query using offset-based pagination.

    Args:
        query: SQLAlchemy query object
        page: Page number (1-based)
        per_page: Number of items per page

    Returns:
        PaginatedResponse containing items and metadata
    """
    return PageBuilder(query, page=page, per_page=per_page).build()


def _encode_cursor(data: Dict[str, Any]) -> str:
    """Encode cursor data to an opaque base64 string."""
    raw = json.dumps(data, default=str).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(cursor: str) -> Optional[Dict[str, Any]]:
    """Decode a base64 cursor string."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode())
        return json.loads(raw)
    except Exception:
        return None


def paginate_with_cursor(
    query,
    cursor: Optional[str] = None,
    per_page: int = DEFAULT_PAGE_SIZE,
    order_column=None,
) -> PaginatedResponse:
    """
    Paginate a SQLAlchemy query using cursor-based pagination.

    Args:
        query: SQLAlchemy query object
        cursor: Opaque cursor string from previous page
        per_page: Number of items per page
        order_column: Column to order by (required for deterministic cursors)

    Returns:
        PaginatedResponse containing items and metadata
    """
    per_page = min(max(MIN_PAGE_SIZE, per_page), MAX_PAGE_SIZE)

    if cursor:
        cursor_data = _decode_cursor(cursor)
        if cursor_data and order_column is not None:
            last_value = cursor_data.get("last_value")
            query = query.filter(order_column > last_value)

    items = query.limit(per_page + 1).all()
    has_next = len(items) > per_page
    if has_next:
        items = items[:per_page]

    next_cursor = None
    if has_next and items and order_column is not None:
        last_item = items[-1]
        col_name = order_column.key
        next_cursor = _encode_cursor({"last_value": getattr(last_item, col_name, None)})

    metadata = PaginationMetadata(
        page=1,
        per_page=per_page,
        total=-1,  # Unknown total in cursor pagination
        pages=-1,
        has_next=has_next,
        has_prev=cursor is not None,
        next_cursor=next_cursor,
    )

    return PaginatedResponse(data=items, pagination=metadata)


def format_pagination_response(
    items: List[Any],
    metadata: PaginationMetadata,
    serialize_fn=None,
) -> Dict[str, Any]:
    """
    Format a paginated response dictionary.

    Args:
        items: List of result items
        metadata: Pagination metadata
        serialize_fn: Optional serializer function for each item

    Returns:
        Formatted response dict
    """
    serialized = [serialize_fn(item) for item in items] if serialize_fn else items
    return PaginatedResponse(data=serialized, pagination=metadata).to_dict()
```
