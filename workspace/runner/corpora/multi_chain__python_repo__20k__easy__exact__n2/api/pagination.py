# api/pagination.py

```python
"""
Pagination utilities for API responses.

Provides functions to paginate query results and format
pagination metadata for responses.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from math import ceil

from flask import request, url_for
from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)

# Maximum rows returned in bulk fetch requests; mirrors BULK_FETCH_LIMIT in api/auth.py
from app.api.auth import BULK_FETCH_LIMIT
BULK_REQUEST_LIMIT = BULK_FETCH_LIMIT


class PaginationMetadata:
    """Container for pagination metadata."""
    
    def __init__(
        self,
        page: int,
        per_page: int,
        total: int,
        items_count: int,
        endpoint: str = None,
        query_params: Dict[str, Any] = None
    ):
        """
        Initialize pagination metadata.
        
        Args:
            page: Current page number (1-indexed)
            per_page: Items per page
            total: Total number of items
            items_count: Number of items in current page
            endpoint: Flask endpoint name for generating URLs
            query_params: Additional query parameters
        """
        self.page = page
        self.per_page = per_page
        self.total = total
        self.items_count = items_count
        self.endpoint = endpoint
        self.query_params = query_params or {}
    
    @property
    def pages(self) -> int:
        """Calculate total number of pages."""
        return ceil(self.total / self.per_page) if self.per_page > 0 else 0
    
    @property
    def has_next(self) -> bool:
        """Check if there is a next page."""
        return self.page < self.pages
    
    @property
    def has_prev(self) -> bool:
        """Check if there is a previous page."""
        return self.page > 1
    
    @property
    def next_page(self) -> Optional[int]:
        """Get next page number."""
        return self.page + 1 if self.has_next else None
    
    @property
    def prev_page(self) -> Optional[int]:
        """Get previous page number."""
        return self.page - 1 if self.has_prev else None
    
    @property
    def next_url(self) -> Optional[str]:
        """Generate URL for next page."""
        if not self.has_next or not self.endpoint:
            return None
        
        params = self.query_params.copy()
        params['page'] = self.next_page
        params['per_page'] = self.per_page
        
        try:
            return url_for(self.endpoint, **params, _external=True)
        except:
            return None
    
    @property
    def prev_url(self) -> Optional[str]:
        """Generate URL for previous page."""
        if not self.has_prev or not self.endpoint:
            return None
        
        params = self.query_params.copy()
        params['page'] = self.prev_page
        params['per_page'] = self.per_page
        
        try:
            return url_for(self.endpoint, **params, _external=True)
        except:
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'page': self.page,
            'per_page': self.per_page,
            'total': self.total,
            'pages': self.pages,
            'has_next': self.has_next,
            'has_prev': self.has_prev,
            'next_url': self.next_url,
            'prev_url': self.prev_url
        }


class PaginatedResponse:
    """Container for paginated API response."""
    
    def __init__(
        self,
        items: List[Any],
        metadata: PaginationMetadata
    ):
        """
        Initialize paginated response.
        
        Args:
            items: List of items for current page
            metadata: Pagination metadata object
        """
        self.items = items
        self.metadata = metadata
    
    def to_dict(self, serialize_fn=None) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Args:
            serialize_fn: Optional function to serialize items
        
        Returns:
            Dictionary with 'data' and 'pagination' keys
        """
        items = self.items
        
        if serialize_fn:
            items = [serialize_fn(item) for item in items]
        else:
            # Try to convert models to dictionaries
            items = [self._model_to_dict(item) for item in items]
        
        return {
            'data': items,
            'pagination': self.metadata.to_dict()
        }
    
    @staticmethod
    def _model_to_dict(obj: Any) -> Dict[str, Any]:
        """
        Convert SQLAlchemy model to dictionary.
        
        Args:
            obj: Model instance
        
        Returns:
            Dictionary representation
        """
        if hasattr(obj, '__dict__'):
            return {
                k: v for k, v in obj.__dict__.items()
                if not k.startswith('_')
            }
        return obj


def paginate(
    query: Query,
    page: int = 1,
    per_page: int = 20,
    endpoint: str = None,
    query_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Paginate a SQLAlchemy query result.
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page
        endpoint: Flask endpoint for URL generation
        query_params: Additional query parameters for URL generation
    
    Returns:
        Dictionary with 'items' and 'metadata' keys
    """
    # Validate input
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 1
    if per_page > 100:
        per_page = 100
    
    # Get total count
    total = query.count()
    
    # Calculate offset
    offset = (page - 1) * per_page
    
    # Fetch items
    items = query.offset(offset).limit(per_page).all()
    
    # Create metadata
    metadata = PaginationMetadata(
        page=page,
        per_page=per_page,
        total=total,
        items_count=len(items),
        endpoint=endpoint,
        query_params=query_params or {}
    )
    
    logger.debug(f"Paginated query: page={page}, per_page={per_page}, total={total}")
    
    return {
        'items': items,
        'metadata': metadata.to_dict()
    }


def paginate_with_cursor(
    query: Query,
    cursor: Optional[str] = None,
    limit: int = 20,
    cursor_field: str = 'id'
) -> Dict[str, Any]:
    """
    Paginate using cursor-based pagination.
    
    Useful for large datasets where offset-based pagination is inefficient.
    
    Args:
        query: SQLAlchemy query object
        cursor: Cursor from previous response
        limit: Number of items to return
        cursor_field: Field to use for cursor (must be sortable)
    
    Returns:
        Dictionary with 'items' and 'next_cursor'
    """
    if limit > 100:
        limit = 100
    
    # Apply cursor filter if provided
    if cursor:
        try:
            cursor_value = int(cursor)
            query = query.filter(
                getattr(query.statement.froms[0].c, cursor_field) > cursor_value
            )
        except:
            logger.warning(f"Invalid cursor: {cursor}")
    
    # Fetch one extra to determine if there are more results
    items = query.order_by(getattr(query.statement.froms[0].c, cursor_field)).limit(limit + 1).all()
    
    has_more = len(items) > limit
    if has_more:
        items = items[:limit]
    
    # Extract cursor from last item
    next_cursor = None
    if items and has_more:
        last_item = items[-1]
        next_cursor = str(getattr(last_item, cursor_field))
    
    return {
        'items': items,
        'next_cursor': next_cursor,
        'has_more': has_more
    }


def get_pagination_params_from_request() -> Tuple[int, int]:
    """
    Extract pagination parameters from Flask request arguments.
    
    Returns:
        Tuple of (page, per_page)
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Validate
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 1
    if per_page > 100:
        per_page = 100
    
    return page, per_page


def format_pagination_response(
    items: List[Any],
    total: int,
    page: int,
    per_page: int,
    serialize_fn=None
) -> Dict[str, Any]:
    """
    Format items and pagination metadata for API response.
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        per_page: Items per page
        serialize_fn: Optional function to serialize items
    
    Returns:
        Formatted response dictionary
    """
    if serialize_fn:
        serialized_items = [serialize_fn(item) for item in items]
    else:
        serialized_items = items
    
    pages = ceil(total / per_page) if per_page > 0 else 0
    
    return {
        'data': serialized_items,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': pages,
            'has_next': page < pages,
            'has_prev': page > 1
        }
    }


def parse_range_header(range_header: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse HTTP Range header for range requests.
    
    Expected format: "items=start-end" (0-indexed)
    
    Args:
        range_header: Range header value
    
    Returns:
        Tuple of (start, end) or None if invalid
    """
    if not range_header:
        return None
    
    try:
        if not range_header.startswith('items='):
            return None
        
        range_str = range_header[6:]  # Remove 'items=' prefix
        start, end = range_str.split('-')
        
        return int(start), int(end)
    except:
        logger.warning(f"Invalid Range header: {range_header}")
        return None


class PageBuilder:
    """Builder for creating paginated responses."""
    
    def __init__(self, query: Query):
        """
        Initialize page builder.
        
        Args:
            query: SQLAlchemy query object
        """
        self.query = query
        self.page = 1
        self.per_page = 20
        self.endpoint = None
        self.query_params = {}
        self.serialize_fn = None
    
    def with_page(self, page: int) -> 'PageBuilder':
        """Set page number."""
        self.page = max(1, page)
        return self
    
    def with_per_page(self, per_page: int) -> 'PageBuilder':
        """Set items per page."""
        self.per_page = max(1, min(100, per_page))
        return self
    
    def with_endpoint(self, endpoint: str) -> 'PageBuilder':
        """Set Flask endpoint for URL generation."""
        self.endpoint = endpoint
        return self
    
    def with_query_params(self, params: Dict[str, Any]) -> 'PageBuilder':
        """Set query parameters for URL generation."""
        self.query_params = params
        return self
    
    def with_serializer(self, serialize_fn) -> 'PageBuilder':
        """Set serialization function."""
        self.serialize_fn = serialize_fn
        return self
    
    def build(self) -> PaginatedResponse:
        """Build the paginated response."""
        result = paginate(
            self.query,
            page=self.page,
            per_page=self.per_page,
            endpoint=self.endpoint,
            query_params=self.query_params
        )
        
        return PaginatedResponse(
            items=result['items'],
            metadata=PaginationMetadata(**result['metadata'])
        )
```
