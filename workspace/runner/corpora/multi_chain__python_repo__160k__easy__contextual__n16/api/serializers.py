# api/serializers.py

```python
"""
Data serialization and deserialization for API requests and responses.

Uses marshmallow schema definitions to convert between database models
and JSON representations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from marshmallow import (
    Schema, fields, post_load, pre_dump, validates,
    ValidationError as MarshmallowValidationError
)
from marshmallow.validate import Length, Email, Range, OneOf

from app.models import User, Project, Task, Organization, Team, Comment
from app.database import db

logger = logging.getLogger(__name__)

SCHEMA_VERSION_TAG = "see VALIDATION_RULESET_ID in api/validators.py"


class BaseSchema(Schema):
    """Base schema with common fields and methods."""
    
    id = fields.Int(dump_only=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    class Meta:
        strict = True


class UserSerializer(BaseSchema):
    """Serializer for User model."""
    
    id = fields.Int()
    email = fields.Email(required=True, validate=Email())
    first_name = fields.Str(validate=Length(min=1, max=100))
    last_name = fields.Str(validate=Length(min=1, max=100))
    full_name = fields.Method('get_full_name', dump_only=True)
    phone = fields.Str(validate=Length(max=20), allow_none=True)
    timezone = fields.Str(allow_none=True)
    role = fields.Str(validate=OneOf(['admin', 'user', 'guest']))
    is_active = fields.Bool()
    is_verified = fields.Bool(dump_only=True)
    last_login = fields.DateTime(dump_only=True, format='iso')
    profile_picture_url = fields.Url(allow_none=True)
    bio = fields.Str(allow_none=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    @staticmethod
    def get_full_name(obj: User) -> str:
        """Get user's full name."""
        return f"{obj.first_name} {obj.last_name}".strip()
    
    @validates('email')
    def validate_email_unique(self, value: str):
        """Validate email is unique."""
        if hasattr(self, 'context') and self.context.get('instance'):
            # Update case - allow same email if updating same user
            existing = User.query.filter(
                User.email == value,
                User.id != self.context['instance'].id
            ).first()
        else:
            # Create case - email must be unique
            existing = User.query.filter_by(email=value).first()
        
        if existing:
            raise MarshmallowValidationError("Email already registered")
    
    class Meta:
        fields = (
            'id', 'email', 'first_name', 'last_name', 'full_name',
            'phone', 'timezone', 'role', 'is_active', 'is_verified',
            'last_login', 'profile_picture_url', 'bio', 'created_at', 'updated_at'
        )


class ProjectSerializer(BaseSchema):
    """Serializer for Project model."""
    
    id = fields.Int()
    name = fields.Str(required=True, validate=Length(min=1, max=200))
    description = fields.Str(allow_none=True, validate=Length(max=2000))
    organization_id = fields.Int(allow_none=True)
    owner_id = fields.Int(dump_only=True)
    owner = fields.Nested(UserSerializer, dump_only=True)
    visibility = fields.Str(
        validate=OneOf(['private', 'internal', 'public']),
        load_default='private'
    )
    status = fields.Str(
        validate=OneOf(['active', 'archived', 'deleted']),
        load_default='active'
    )
    task_count = fields.Method('get_task_count', dump_only=True)
    member_count = fields.Method('get_member_count', dump_only=True)
    repository_url = fields.Url(allow_none=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    @staticmethod
    def get_task_count(obj: Project) -> int:
        """Get number of tasks in project."""
        return len(obj.tasks) if hasattr(obj, 'tasks') else 0
    
    @staticmethod
    def get_member_count(obj: Project) -> int:
        """Get number of members in project."""
        return len(obj.members) if hasattr(obj, 'members') else 0
    
    class Meta:
        fields = (
            'id', 'name', 'description', 'organization_id', 'owner_id',
            'owner', 'visibility', 'status', 'task_count', 'member_count',
            'repository_url', 'created_at', 'updated_at'
        )


class TaskSerializer(BaseSchema):
    """Serializer for Task model."""
    
    id = fields.Int()
    title = fields.Str(required=True, validate=Length(min=1, max=300))
    description = fields.Str(allow_none=True, validate=Length(max=5000))
    project_id = fields.Int(required=True)
    project = fields.Nested(ProjectSerializer, dump_only=True)
    created_by = fields.Int(dump_only=True)
    creator = fields.Nested(UserSerializer, dump_only=True)
    assigned_to = fields.Int(allow_none=True)
    assignee = fields.Nested(UserSerializer, allow_none=True, dump_only=True)
    status = fields.Str(
        validate=OneOf(['open', 'in_progress', 'review', 'completed', 'closed']),
        load_default='open'
    )
    priority = fields.Str(
        validate=OneOf(['low', 'medium', 'high', 'critical']),
        load_default='medium'
    )
    due_date = fields.DateTime(allow_none=True, format='iso')
    estimated_hours = fields.Float(allow_none=True, validate=Range(min=0.25))
    actual_hours = fields.Float(allow_none=True, validate=Range(min=0))
    labels = fields.List(fields.Str())
    is_blocked = fields.Bool(dump_only=True)
    blocked_by_ids = fields.List(fields.Int(), dump_only=True)
    comment_count = fields.Method('get_comment_count', dump_only=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    @staticmethod
    def get_comment_count(obj: Task) -> int:
        """Get number of comments on task."""
        return len(obj.comments) if hasattr(obj, 'comments') else 0
    
    class Meta:
        fields = (
            'id', 'title', 'description', 'project_id', 'project',
            'created_by', 'creator', 'assigned_to', 'assignee',
            'status', 'priority', 'due_date', 'estimated_hours',
            'actual_hours', 'labels', 'is_blocked', 'blocked_by_ids',
            'comment_count', 'created_at', 'updated_at'
        )


class CommentSerializer(BaseSchema):
    """Serializer for Comment model."""
    
    id = fields.Int()
    content = fields.Str(required=True, validate=Length(min=1, max=3000))
    task_id = fields.Int(required=True)
    author_id = fields.Int(dump_only=True)
    author = fields.Nested(UserSerializer, dump_only=True)
    is_edited = fields.Bool(dump_only=True)
    edited_at = fields.DateTime(allow_none=True, dump_only=True, format='iso')
    mentions = fields.List(fields.Int(), dump_only=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    class Meta:
        fields = (
            'id', 'content', 'task_id', 'author_id', 'author',
            'is_edited', 'edited_at', 'mentions', 'created_at', 'updated_at'
        )


class OrganizationSerializer(BaseSchema):
    """Serializer for Organization model."""
    
    id = fields.Int()
    name = fields.Str(required=True, validate=Length(min=1, max=200))
    description = fields.Str(allow_none=True, validate=Length(max=2000))
    website = fields.Url(allow_none=True)
    logo_url = fields.Url(allow_none=True)
    owner_id = fields.Int(dump_only=True)
    owner = fields.Nested(UserSerializer, dump_only=True)
    is_public = fields.Bool(load_default=False)
    member_count = fields.Method('get_member_count', dump_only=True)
    project_count = fields.Method('get_project_count', dump_only=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    @staticmethod
    def get_member_count(obj: Organization) -> int:
        """Get number of members."""
        return len(obj.members) if hasattr(obj, 'members') else 0
    
    @staticmethod
    def get_project_count(obj: Organization) -> int:
        """Get number of projects."""
        return len(obj.projects) if hasattr(obj, 'projects') else 0
    
    class Meta:
        fields = (
            'id', 'name', 'description', 'website', 'logo_url',
            'owner_id', 'owner', 'is_public', 'member_count',
            'project_count', 'created_at', 'updated_at'
        )


class TeamSerializer(BaseSchema):
    """Serializer for Team model."""
    
    id = fields.Int()
    name = fields.Str(required=True, validate=Length(min=1, max=100))
    description = fields.Str(allow_none=True, validate=Length(max=1000))
    organization_id = fields.Int(required=True)
    organization = fields.Nested(OrganizationSerializer, dump_only=True)
    lead_id = fields.Int(allow_none=True)
    lead = fields.Nested(UserSerializer, allow_none=True, dump_only=True)
    member_count = fields.Method('get_member_count', dump_only=True)
    created_at = fields.DateTime(dump_only=True, format='iso')
    updated_at = fields.DateTime(dump_only=True, format='iso')
    
    @staticmethod
    def get_member_count(obj: Team) -> int:
        """Get number of team members."""
        return len(obj.members) if hasattr(obj, 'members') else 0
    
    class Meta:
        fields = (
            'id', 'name', 'description', 'organization_id', 'organization',
            'lead_id', 'lead', 'member_count', 'created_at', 'updated_at'
        )


class ErrorSerializer(Schema):
    """Serializer for error responses."""
    
    error = fields.Str(required=True)
    message = fields.Str(allow_none=True)
    status_code = fields.Int(required=True)
    request_id = fields.Str(dump_only=True)
    timestamp = fields.DateTime(dump_only=True, format='iso')
    details = fields.Dict(allow_none=True)
    
    class Meta:
        fields = ('error', 'message', 'status_code', 'request_id', 'timestamp', 'details')


class PaginationMetadataSerializer(Schema):
    """Serializer for pagination metadata."""
    
    page = fields.Int()
    per_page = fields.Int()
    total = fields.Int()
    pages = fields.Int()
    has_next = fields.Bool()
    has_prev = fields.Bool()
    next_url = fields.Url(allow_none=True)
    prev_url = fields.Url(allow_none=True)
    
    class Meta:
        fields = (
            'page', 'per_page', 'total', 'pages',
            'has_next', 'has_prev', 'next_url', 'prev_url'
        )


class PaginatedResponseSerializer(Schema):
    """Serializer for paginated responses."""
    
    data = fields.List(fields.Dict())
    pagination = fields.Nested(PaginationMetadataSerializer)
    
    class Meta:
        fields = ('data', 'pagination')


# Alias common serializers for easier imports
UserSchema = UserSerializer
ProjectSchema = ProjectSerializer
TaskSchema = TaskSerializer
CommentSchema = CommentSerializer
OrganizationSchema = OrganizationSerializer
TeamSchema = TeamSerializer
ErrorSchema = ErrorSerializer
```
