# api/serializers.py

```python
"""
Data serialization schemas for API responses.

Uses marshmallow for schema-based serialization and deserialization
of SQLAlchemy models. Provides consistent response shapes across
all API endpoints.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_serializers = "pipeline_error_handlers"


class BaseSerializer:
    """Base class for all serializers."""

    fields: List[str] = []

    @classmethod
    def dump(cls, obj: Any) -> Dict[str, Any]:
        """
        Serialize a single object.

        Args:
            obj: Model instance to serialize

        Returns:
            Dictionary representation
        """
        result = {}
        for field_name in cls.fields:
            value = getattr(obj, field_name, None)
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            result[field_name] = value
        return result

    @classmethod
    def dump_many(cls, objects: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize multiple objects.

        Args:
            objects: List of model instances

        Returns:
            List of dictionary representations
        """
        return [cls.dump(obj) for obj in objects]


class UserSerializer(BaseSerializer):
    """Serializer for User model."""

    fields = [
        "id",
        "email",
        "username",
        "full_name",
        "bio",
        "avatar_url",
        "is_active",
        "is_verified",
        "created_at",
        "updated_at",
        "last_login_at",
        "organization_id",
    ]

    @classmethod
    def dump(cls, obj: Any) -> Dict[str, Any]:
        data = super().dump(obj)
        # Never expose hashed_password
        data.pop("hashed_password", None)
        return data


class ProjectSerializer(BaseSerializer):
    """Serializer for Project model."""

    fields = [
        "id",
        "name",
        "description",
        "visibility",
        "status",
        "organization_id",
        "owner_id",
        "created_at",
        "updated_at",
    ]


class TaskSerializer(BaseSerializer):
    """Serializer for Task model."""

    fields = [
        "id",
        "title",
        "description",
        "status",
        "priority",
        "estimated_hours",
        "user_id",
        "project_id",
        "assigned_to",
        "organization_id",
        "created_at",
        "updated_at",
        "completed_at",
    ]


class CommentSerializer(BaseSerializer):
    """Serializer for TaskComment model."""

    fields = [
        "id",
        "task_id",
        "user_id",
        "body",
        "created_at",
        "updated_at",
    ]


class OrganizationSerializer(BaseSerializer):
    """Serializer for Organization model."""

    fields = [
        "id",
        "name",
        "slug",
        "description",
        "is_active",
        "created_at",
        "updated_at",
    ]


class TeamSerializer(BaseSerializer):
    """Serializer for Team model."""

    fields = [
        "id",
        "name",
        "organization_id",
        "created_at",
        "updated_at",
    ]


class ErrorSerializer:
    """Serializer for API error responses."""

    @staticmethod
    def dump(error_code: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Serialize an error response.

        Args:
            error_code: Machine-readable error code
            message: Human-readable message
            details: Optional additional context

        Returns:
            Error response dictionary
        """
        result: Dict[str, Any] = {
            "error": error_code,
            "message": message,
        }
        if details:
            result["details"] = details
        return result


class PaginationMetadataSerializer:
    """Serializer for pagination metadata."""

    @staticmethod
    def dump(metadata) -> Dict[str, Any]:
        """
        Serialize pagination metadata.

        Args:
            metadata: PaginationMetadata instance

        Returns:
            Serialized metadata dictionary
        """
        return {
            "page": metadata.page,
            "per_page": metadata.per_page,
            "total": metadata.total,
            "pages": metadata.pages,
            "has_next": metadata.has_next,
            "has_prev": metadata.has_prev,
        }
```
