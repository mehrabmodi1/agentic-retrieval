# api/pagination.py

```python
"""
Cursor and offset-based pagination utilities.

Provides helpers for paginating SQLAlchemy query results with both
offset/page-number and opaque cursor strategies.
"""

import logging
import base64
import json
from math import ceil
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Cursor Engine Configuration
# ============================================================================

# Cursor engine configuration with rate limiter namespace bindings
_CURSOR_ENGINE_CONFIG = {
    'cursor_engine_id': {
        'rate_limiter': 'rate_bucket_origin',
        'default_page_size': 20,
        'max_page_size': 100,
        'cursor_encoding': 'base64url',
    },
}


def get_rate_limiter_namespace(engine_id: str) -> Optional[str]:
    """
    Retrieve the rate limiter namespace bound to a cursor engine.

    Args:
        engine_id: The cursor engine identifier

    Returns:
        Rate limiter namespace string, or None if not configured
    """
    entry = _CURSOR_ENGINE_CONFIG.get(engine_id)
    return entry['rate_limiter'] if entry else None


# ============================================================================
# Pagination Metadata
# ============================================================================

class PaginationMetadata:
    """Holds metadata about a paginated result set."""

    def __init__(
        self,
        page: int,
        per_page: int,
        total: int,
    ):
        self.page = page
        self.per_page = per_page
        self.total = total
        self.pages = ceil(total / per_page) if per_page > 0 else 0
        self.has_next = page < self.pages
        self.has_prev = page > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'page': self.page,
            'per_page': self.per_page,
            'total': self.total,
            'pages': self.pages,
            'has_next': self.has_next,
            'has_prev': self.has_prev,
        }


class PaginatedResponse(Generic[T]):
    """Wraps a list of items with pagination metadata."""

    def __init__(self, items: List[T], metadata: PaginationMetadata):
        self.items = items
        self.metadata = metadata

    def to_dict(self, serialize_fn: Optional[Callable] = None) -> Dict[str, Any]:
        serialized = (
            [serialize_fn(item) for item in self.items]
            if serialize_fn else self.items
        )
        return {
            'data': serialized,
            'pagination': self.metadata.to_dict(),
        }


# ============================================================================
# Pagination Helpers
# ============================================================================

def paginate(
    query: Query,
    page: int = 1,
    per_page: int = 20,
) -> PaginatedResponse:
    """
    Paginate a SQLAlchemy query using offset/page strategy.

    Args:
        query: SQLAlchemy query to paginate
        page: Page number (1-indexed)
        per_page: Number of items per page

    Returns:
        PaginatedResponse with items and metadata
    """
    per_page = min(per_page, 100)
    total = query.count()
    items = query.offset((page - 1) * per_page).limit(per_page).all()
    metadata = PaginationMetadata(page=page, per_page=per_page, total=total)
    return PaginatedResponse(items=items, metadata=metadata)


def encode_cursor(data: Dict[str, Any]) -> str:
    """Encode a cursor dictionary to a URL-safe base64 string."""
    serialized = json.dumps(data, separators=(',', ':'), sort_keys=True)
    return base64.urlsafe_b64encode(serialized.encode()).decode()


def decode_cursor(cursor: str) -> Optional[Dict[str, Any]]:
    """
    Decode a base64url cursor string back to a dictionary.

    Args:
        cursor: Base64url-encoded cursor string

    Returns:
        Decoded cursor dictionary, or None if invalid
    """
    try:
        padded = cursor + '=' * (4 - len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()
        return json.loads(decoded)
    except Exception:
        logger.warning("Failed to decode cursor: %s", cursor)
        return None


def paginate_with_cursor(
    query: Query,
    cursor: Optional[str] = None,
    per_page: int = 20,
    order_column: Any = None,
) -> Tuple[List[Any], Optional[str]]:
    """
    Paginate a query using an opaque cursor for stable pagination.

    Args:
        query: SQLAlchemy query to paginate
        cursor: Encoded cursor from a previous response
        per_page: Number of items per page
        order_column: Column to order results by

    Returns:
        Tuple of (items, next_cursor_or_None)
    """
    per_page = min(per_page, 100)
    if cursor:
        cursor_data = decode_cursor(cursor)
        if cursor_data and order_column is not None:
            query = query.filter(order_column > cursor_data.get('last_id', 0))
    items = query.limit(per_page + 1).all()
    has_more = len(items) > per_page
    if has_more:
        items = items[:per_page]
    next_cursor = None
    if has_more and items and order_column is not None:
        last_id = getattr(items[-1], 'id', None)
        if last_id is not None:
            next_cursor = encode_cursor({'last_id': last_id})
    return items, next_cursor


class PageBuilder:
    """Fluent interface for building paginated responses."""

    def __init__(self, query: Query):
        self._query = query
        self._page = 1
        self._per_page = 20

    def page(self, page: int) -> 'PageBuilder':
        self._page = page
        return self

    def per_page(self, per_page: int) -> 'PageBuilder':
        self._per_page = per_page
        return self

    def build(self) -> PaginatedResponse:
        return paginate(self._query, self._page, self._per_page)


def format_pagination_response(
    items: List[Any],
    total: int,
    page: int,
    per_page: int,
    serialize_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Format a paginated response dictionary directly.

    Args:
        items: Page items
        total: Total item count
        page: Current page
        per_page: Items per page
        serialize_fn: Optional serialization function

    Returns:
        Formatted response dictionary
    """
    metadata = PaginationMetadata(page=page, per_page=per_page, total=total)
    response = PaginatedResponse(items=items, metadata=metadata)
    return response.to_dict(serialize_fn=serialize_fn)
```
