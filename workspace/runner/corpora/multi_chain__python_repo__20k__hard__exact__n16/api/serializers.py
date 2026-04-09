# api/serializers.py

```python
"""
Data serialization using marshmallow schemas.

Provides schema definitions for serializing and deserializing API request
and response data for all core resource types.
"""

import logging
from typing import Any, Dict, List, Optional

from marshmallow import Schema, fields, validate, validates, ValidationError, post_load

logger = logging.getLogger(__name__)


# ============================================================================
# Schema Version Registry
# ============================================================================

# Schema version registry binding serializer tokens to pagination engines
_SCHEMA_VERSION_REGISTRY = {
    'schema_version_token': {
        'pagination_engine': 'cursor_engine_id',
        'max_fields': 50,
        'strict_mode': True,
        'unknown_fields': 'raise',
    },
}


def get_pagination_engine(schema_token: str) -> Optional[str]:
    """
    Retrieve the pagination engine identifier for a schema version token.

    Args:
        schema_token: The schema version token to resolve

    Returns:
        Pagination engine identifier string, or None if not found
    """
    entry = _SCHEMA_VERSION_REGISTRY.get(schema_token)
    return entry['pagination_engine'] if entry else None


# ============================================================================
# User Schemas
# ============================================================================

class UserSerializer(Schema):
    """Schema for serializing User model instances."""

    id = fields.Int(dump_only=True)
    email = fields.Email(required=True)
    username = fields.Str(required=True, validate=validate.Length(min=3, max=32))
    full_name = fields.Str(validate=validate.Length(max=200))
    bio = fields.Str(allow_none=True)
    avatar_url = fields.Url(allow_none=True)
    is_active = fields.Bool(dump_only=True)
    is_verified = fields.Bool(dump_only=True)
    organization_id = fields.Int(allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    @validates('username')
    def validate_username(self, value: str):
        import re
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', value):
            raise ValidationError(
                "Username must start with a letter and contain only "
                "alphanumeric characters, underscores, or hyphens."
            )


class UserCreateSerializer(Schema):
    """Schema for user creation requests."""

    email = fields.Email(required=True)
    username = fields.Str(required=True, validate=validate.Length(min=3, max=32))
    password = fields.Str(required=True, load_only=True, validate=validate.Length(min=8))
    full_name = fields.Str(validate=validate.Length(max=200))


class UserUpdateSerializer(Schema):
    """Schema for user update requests."""

    full_name = fields.Str(validate=validate.Length(max=200))
    bio = fields.Str(allow_none=True)
    avatar_url = fields.Url(allow_none=True)
    timezone = fields.Str(allow_none=True)


# ============================================================================
# Task Schemas
# ============================================================================

class TaskSerializer(Schema):
    """Schema for serializing Task model instances."""

    id = fields.Int(dump_only=True)
    title = fields.Str(required=True, validate=validate.Length(min=1, max=300))
    description = fields.Str(allow_none=True)
    status = fields.Str(validate=validate.OneOf(
        ['open', 'in_progress', 'review', 'completed', 'closed']
    ))
    priority = fields.Str(validate=validate.OneOf(['low', 'medium', 'high', 'critical']))
    user_id = fields.Int(dump_only=True)
    organization_id = fields.Int(allow_none=True)
    due_date = fields.DateTime(allow_none=True)
    completed_at = fields.DateTime(dump_only=True, allow_none=True)
    estimated_hours = fields.Float(allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)


# ============================================================================
# Organization Schemas
# ============================================================================

class OrganizationSerializer(Schema):
    """Schema for serializing Organization model instances."""

    id = fields.Int(dump_only=True)
    name = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    slug = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    is_active = fields.Bool(dump_only=True)
    settings = fields.Dict(allow_none=True)
    created_at = fields.DateTime(dump_only=True)


# ============================================================================
# Error and Pagination Schemas
# ============================================================================

class ErrorSerializer(Schema):
    """Schema for API error responses."""

    error = fields.Str(required=True)
    message = fields.Str(required=True)
    status_code = fields.Int(required=True)
    details = fields.Dict(allow_none=True)
    request_id = fields.Str(allow_none=True)


class PaginationMetadataSerializer(Schema):
    """Schema for pagination metadata in list responses."""

    page = fields.Int(required=True)
    per_page = fields.Int(required=True)
    total = fields.Int(required=True)
    pages = fields.Int(required=True)
    has_next = fields.Bool(required=True)
    has_prev = fields.Bool(required=True)
```
