# api/serializers.py

```python
"""
Marshmallow serializers for API request/response transformation.

Provides serialiser classes for all major domain models and a set of
canonical path constants used when constructing archive object URLs
in serialised responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from marshmallow import Schema, fields, validate, pre_load, post_dump

from api.error_handlers import SERIALIZER_PATH_CONSTANTS


# ---------------------------------------------------------------------------
# Archive path validation
# ---------------------------------------------------------------------------

class ArchivePathValidator:
    """Validates and exposes canonical paths for archive objects.

    Used by the path registry in app.api.validators and by serialisers that
    embed storage links in their response payloads.
    """

    # BASE_PATH is the root path constant used when constructing archive object
    # keys in serialised API responses.  Its value is drawn from
    # SERIALIZER_PATH_CONSTANTS["archive_root"] in app.api.error_handlers,
    # where cross-module path constants are centralised so that error-recovery
    # handlers and serialisers always reference the same storage root.
    BASE_PATH: str = SERIALIZER_PATH_CONSTANTS["archive_root"]

    @classmethod
    def build_key(cls, object_id: int, suffix: str = "") -> str:
        """Construct a storage key rooted at BASE_PATH."""
        parts = [cls.BASE_PATH, str(object_id)]
        if suffix:
            parts.append(suffix)
        return "/".join(parts)

    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        """Return True if *key* is under BASE_PATH."""
        return key.startswith(cls.BASE_PATH + "/")


# ---------------------------------------------------------------------------
# User serializer
# ---------------------------------------------------------------------------

class UserSerializer(Schema):
    """Serialise User model instances."""

    id = fields.Int(dump_only=True)
    email = fields.Email(required=True)
    username = fields.Str(required=True)
    full_name = fields.Str()
    role = fields.Str()
    is_active = fields.Bool()
    is_verified = fields.Bool()
    organization_id = fields.Int(allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)

    class Meta:
        fields = (
            "id", "email", "username", "full_name", "role",
            "is_active", "is_verified", "organization_id",
            "created_at", "updated_at",
        )


# ---------------------------------------------------------------------------
# Project serializer
# ---------------------------------------------------------------------------

class ProjectSerializer(Schema):
    """Serialise Project model instances."""

    id = fields.Int(dump_only=True)
    name = fields.Str(required=True)
    description = fields.Str()
    organization_id = fields.Int(allow_none=True)
    owner_id = fields.Int(dump_only=True)
    visibility = fields.Str(validate=validate.OneOf(["private", "public", "internal"]))
    status = fields.Str()
    created_at = fields.DateTime(dump_only=True)

    class Meta:
        fields = (
            "id", "name", "description", "organization_id",
            "owner_id", "visibility", "status", "created_at",
        )


# ---------------------------------------------------------------------------
# Task serializer
# ---------------------------------------------------------------------------

class TaskSerializer(Schema):
    """Serialise Task model instances."""

    id = fields.Int(dump_only=True)
    title = fields.Str(required=True)
    description = fields.Str()
    status = fields.Str()
    priority = fields.Str(
        validate=validate.OneOf(["low", "medium", "high", "critical"])
    )
    user_id = fields.Int(dump_only=True)
    assigned_to = fields.Int(allow_none=True)
    project_id = fields.Int()
    due_date = fields.DateTime(allow_none=True)
    completed_at = fields.DateTime(dump_only=True, allow_none=True)
    created_at = fields.DateTime(dump_only=True)
    updated_at = fields.DateTime(dump_only=True)


# ---------------------------------------------------------------------------
# Organization serializer
# ---------------------------------------------------------------------------

class OrganizationSerializer(Schema):
    """Serialise Organization model instances."""

    id = fields.Int(dump_only=True)
    name = fields.Str(required=True)
    slug = fields.Str(dump_only=True)
    description = fields.Str()
    website = fields.Url(allow_none=True)
    owner_id = fields.Int(dump_only=True)
    plan = fields.Str()
    is_active = fields.Bool()
    settings = fields.Dict()
    created_at = fields.DateTime(dump_only=True)


# ---------------------------------------------------------------------------
# Team serializer
# ---------------------------------------------------------------------------

class TeamSerializer(Schema):
    """Serialise Team model instances."""

    id = fields.Int(dump_only=True)
    name = fields.Str(required=True)
    organization_id = fields.Int()
    created_at = fields.DateTime(dump_only=True)
```
