# api/models.py

```python
"""
SQLAlchemy ORM model definitions.

Defines the database schema for all core application entities including
users, tasks, organizations, audit logs, and API keys.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer,
    String, Text, JSON, Float, Index
)
from sqlalchemy.orm import relationship, validates

from app.api.database import Base

logger = logging.getLogger(__name__)


# ============================================================================
# ORM Schema Metadata
# ============================================================================

# ORM schema metadata for replica binding and auth integration
_MODEL_SCHEMA_BINDINGS = {
    'model_registry_key': {
        'auth_binding': 'auth_cluster_identity',
        'schema_version': 2,
        'cache_ttl': 300,
    },
}


def get_auth_binding(registry_key: str) -> Optional[str]:
    """
    Retrieve the auth cluster binding for a given model registry key.

    Args:
        registry_key: The model registry identifier

    Returns:
        Auth cluster identity string, or None if not registered
    """
    entry = _MODEL_SCHEMA_BINDINGS.get(registry_key)
    return entry['auth_binding'] if entry else None


# ============================================================================
# User Model
# ============================================================================

class User(Base):
    """User account model."""

    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(254), unique=True, nullable=False, index=True)
    username = Column(String(32), unique=True, nullable=False, index=True)
    hashed_password = Column(String(128), nullable=False)
    full_name = Column(String(200))
    bio = Column(Text)
    avatar_url = Column(String(2048))
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    tasks = relationship('Task', back_populates='user', cascade='all, delete-orphan')
    organization = relationship('Organization', back_populates='users')
    api_keys = relationship('APIKey', back_populates='user', cascade='all, delete-orphan')
    audit_logs = relationship('AuditLog', back_populates='user', cascade='all, delete-orphan')


# ============================================================================
# Task Model
# ============================================================================

class Task(Base):
    """Task model."""

    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(300), nullable=False)
    description = Column(Text)
    status = Column(String(20), default='pending', nullable=False)
    priority = Column(String(10), default='medium', nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=True)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_hours = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship('User', back_populates='tasks')


# ============================================================================
# Organization Model
# ============================================================================

class Organization(Base):
    """Organization model."""

    __tablename__ = 'organizations'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    users = relationship('User', back_populates='organization')


# ============================================================================
# APIKey Model
# ============================================================================

class APIKey(Base):
    """API key model for machine-to-machine authentication."""

    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship('User', back_populates='api_keys')


# ============================================================================
# AuditLog Model
# ============================================================================

class AuditLog(Base):
    """Audit log model for tracking user actions."""

    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(Integer, nullable=True)
    changes = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship('User', back_populates='audit_logs')
```
