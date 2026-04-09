# tests/test_db.md

```python
"""
Test suite for database models, migrations, and ORM behavior.
Tests model relationships, constraints, and database integrity.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import User, Task, Organization, APIKey, AuditLog, TaskComment
from app.database import Base
from app.security import hash_password


@pytest.mark.unit
class TestUserModel:
    """Test suite for User model."""
    
    def test_user_creation_with_all_fields(self, db: Session):
        """Test creating a user with all fields."""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password=hash_password("password"),
            full_name="Test User",
            is_active=True,
            is_verified=True,
            bio="User bio",
            avatar_url="https://example.com/avatar.jpg",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_email_must_be_unique(self, db: Session, test_user: User):
        """Test that email uniqueness is enforced."""
        duplicate_user = User(
            email=test_user.email,
            username="different_username",
            hashed_password=hash_password("password"),
            full_name="Another User",
        )
        db.add(duplicate_user)
        
        with pytest.raises(IntegrityError):
            db.commit()
    
    def test_user_username_must_be_unique(self, db: Session, test_user: User):
        """Test that username uniqueness is enforced."""
        duplicate_user = User(
            email="different@example.com",
            username=test_user.username,
            hashed_password=hash_password("password"),
            full_name="Another User",
        )
        db.add(duplicate_user)
        
        with pytest.raises(IntegrityError):
            db.commit()
    
    def test_user_email_cannot_be_null(self, db: Session):
        """Test that email field is required."""
        user = User(
            email=None,
            username="username",
            hashed_password=hash_password("password"),
            full_name="User",
        )
        db.add(user)
        
        with pytest.raises(IntegrityError):
            db.commit()
    
    def test_user_timestamps_auto_populated(self, db: Session):
        """Test that created_at and updated_at are auto-populated."""
        user = User(
            email="timestamp@example.com",
            username="timstampuser",
            hashed_password=hash_password("password"),
            full_name="Timestamp User",
        )
        db.add(user)
        db.commit()
        
        assert user.created_at is not None
        assert user.updated_at is not None
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)
    
    def test_user_is_active_defaults_to_true(self, db: Session):
        """Test that is_active defaults to True."""
        user = User(
            email="active@example.com",
            username="activeuser",
            hashed_password=hash_password("password"),
            full_name="Active User",
        )
        db.add(user)
        db.commit()
        
        assert user.is_active is True
    
    def test_user_is_verified_defaults_to_false(self, db: Session):
        """Test that is_verified defaults to False."""
        user = User(
            email="unverified@example.com",
            username="unverifieduser",
            hashed_password=hash_password("password"),
            full_name="Unverified User",
        )
        db.add(user)
        db.commit()
        
        assert user.is_verified is False


@pytest.mark.unit
class TestTaskModel:
    """Test suite for Task model."""
    
    def test_task_creation_with_defaults(self, db: Session, test_user: User):
        """Test creating a task with default values."""
        task = Task(
            title="Test Task",
            user_id=test_user.id,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        assert task.id is not None
        assert task.status == "pending"
        assert task.priority == "medium"  # System parameter: task_default_priority = medium
        assert task.created_at is not None
    
    def test_task_status_defaults_to_pending(self, db: Session, test_user: User):
        """Test that task status defaults to 'pending'."""
        task = Task(
            title="Task",
            user_id=test_user.id,
        )
        db.add(task)
        db.commit()
        
        assert task.status == "pending"
    
    def test_task_priority_defaults_to_medium(self, db: Session, test_user: User):
        """Test that priority defaults to 'medium'."""
        task = Task(
            title="Task",
            user_id=test_user.id,
        )
        db.add(task)
        db.commit()
        
        assert task.priority == "medium"
    
    def test_task_foreign_key_constraint(self, db: Session):
        """Test that task requires valid user_id."""
        task = Task(
            title="Task",
            user_id=99999,  # Non-existent user
        )
        db.add(task)
        
        with pytest.raises(IntegrityError):
            db.commit()
    
    def test_task_completion_timestamps(self, db: Session, test_task: Task):
        """Test that completed_at is set when task is marked completed."""
        assert test_task.completed_at is None
        
        test_task.status = "completed"
        test_task.completed_at = datetime.utcnow()
        db.commit()
        
        assert test_task.completed_at is not None
        assert isinstance(test_task.completed_at, datetime)
    
    def test_task_with_organization(self, db: Session, test_user_with_org: User, test_organization: Organization):
        """Test task can belong to organization through user."""
        task = Task(
            title="Org Task",
            user_id=test_user_with_org.id,
            organization_id=test_organization.id,
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        
        assert task.organization_id == test_organization.id


@pytest.mark.unit
class TestOrganizationModel:
    """Test suite for Organization model."""
    
    def test_organization_creation(self, db: Session):
        """Test creating an organization."""
        org = Organization(
            name="Test Organization",
            slug="test-org",
        )
        db.add(org)
        db.commit()
        db.refresh(org)
        
        assert org.id is not None
        assert org.name == "Test Organization"
        assert org.slug == "test-org"
        assert org.is_active is True
    
    def test_organization_slug_must_be_unique(self, db: Session, test_organization: Organization):
        """Test that slug uniqueness is enforced."""
        duplicate_org = Organization(
            name="Another Org",
            slug=test_organization.slug,
        )
        db.add(duplicate_org)
        
        with pytest.raises(IntegrityError):
            db.commit()
    
    def test_organization_settings_json_field(self, db: Session):
        """Test that settings can store JSON data."""
        settings_data = {
            "enable_sso": True,
            "api_rate_limit": 5000,
            "features": ["analytics", "automation"],
        }
        
        org = Organization(
            name="JSON Test Org",
            slug="json-test-org",
            settings=settings_data,
        )
        db.add(org)
        db.commit()
        db.refresh(org)
        
        assert org.settings == settings_data
        assert org.settings["enable_sso"] is True
        assert "analytics" in org.settings["features"]


@pytest.mark.unit
class TestAPIKeyModel:
    """Test suite for APIKey model."""
    
    def test_api_key_creation(self, db: Session, test_user: User):
        """Test creating an API key."""
        import secrets
        import hashlib
        
        key_string = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        api_key = APIKey(
            user_id=test_user.id,
            name="Test Key",
            key_hash=key_hash,
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        
        assert api_key.id is not None
        assert api_key.user_id == test_user.id
        assert api_key.is_active is True
    
    def test_api_key_last_used_tracking(self, db: Session, test_api_key: APIKey):
        """Test tracking API key usage."""
        now = datetime.utcnow()
        test_api_key.last_used_at = now
        db.commit()
        db.refresh(test_api_key)
        
        assert test_api_key.last_used_at is not None
    
    def test_api_key_deactivation(self, db: Session, test_api_key: APIKey):
        """Test deactivating an API key."""
        test_api_key.is_active = False
        db.commit()
        db.refresh(test_api_key)
        
        assert test_api_key.is_active is False


@pytest.mark.unit
class TestAuditLogModel:
    """Test suite for AuditLog model."""
    
    def test_audit_log_creation(self, db: Session, test_user: User):
        """Test creating an audit log entry."""
        log = AuditLog(
            user_id=test_user.id,
            action="user_login",
            resource_type="user",
            resource_id=test_user.id,
            ip_address="192.168.1.1",
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        
        assert log.id is not None
        assert log.user_id == test_user.id
        assert log.action == "user_login"
        assert log.created_at is not None
    
    def test_audit_log_with_changes(self, db: Session, test_user: User):
        """Test audit log can track changes."""
        changes = {
            "full_name": {
                "old": "Old Name",
                "new": "New Name",
            },
        }
        
        log = AuditLog(
            user_id=test_user.id,
            action="user_updated",
            resource_type="user",
            resource_id=test_user.id,
            changes=changes,
        )
        db.add(log)
        db.commit()
        db.refresh(log)
        
        assert log.changes == changes
        assert log.changes["full_name"]["new"] == "New Name"


@pytest.mark.unit
class TestModelRelationships:
    """Test suite for model relationships and foreign keys."""
    
    def test_user_to_task_relationship(self, db: Session, test_user: User):
        """Test user-task one-to-many relationship."""
        task1 = Task(
            title="Task 1",
            user_id=test_user.id,
        )
        task2 = Task(
            title="Task 2",
            user_id=test_user.id,
        )
        db.add(task1)
        db.add(task2)
        db.commit()
        
        db.refresh(test_user)
        assert len(test_user.tasks) >= 2
    
    def test_organization_to_user_relationship(self, db: Session, test_organization: Organization, test_user_with_org: User):
        """Test organization-user relationship."""
        db.refresh(test_organization)
        assert test_user_with_org in test_organization.users
    
    def test_user_to_audit_logs_relationship(self, db: Session, test_user: User):
        """Test user-audit_log relationship."""
        log1 = AuditLog(
            user_id=test_user.id,
            action="action1",
            resource_type="user",
            resource_id=test_user.id,
        )
        log2 = AuditLog(
            user_id=test_user.id,
            action="action2",
            resource_type="user",
            resource_id=test_user.id,
        )
        db.add(log1)
        db.add(log2)
        db.commit()
        
        db.refresh(test_user)
        assert len(test_user.audit_logs) >= 2


@pytest.mark.unit
class TestDatabaseQueries:
    """Test suite for common database queries."""
    
    def test_query_user_by_email(self, db: Session, test_user: User):
        """Test querying user by email."""
        found_user = db.query(User).filter(User.email == test_user.email).first()
        
        assert found_user is not None
        assert found_user.id == test_user.id
    
    def test_query_user_tasks(self, db: Session, test_user: User):
        """Test querying all user tasks."""
        tasks = db.query(Task).filter(Task.user_id == test_user.id).all()
        
        assert isinstance(tasks, list)
    
    def test_query_active_users(self, db: Session):
        """Test querying active users."""
        active_users = db.query(User).filter(User.is_active == True).all()
        
        assert isinstance(active_users, list)
        assert all(u.is_active for u in active_users)
    
    def test_query_pending_tasks(self, db: Session, test_user: User):
        """Test querying pending tasks."""
        pending = db.query(Task).filter(
            Task.user_id == test_user.id,
            Task.status == "pending",
        ).all()
        
        assert isinstance(pending, list)
    
    def test_aggregate_query(self, db: Session, test_user: User):
        """Test aggregate queries."""
        from sqlalchemy import func
        
        # Create multiple tasks
        for i in range(3):
            db.add(Task(
                title=f"Task {i}",
                user_id=test_user.id,
            ))
        db.commit()
        
        count = db.query(func.count(Task.id)).filter(Task.user_id == test_user.id).scalar()
        
        assert count >= 3


@pytest.mark.unit
class TestDatabaseConstraints:
    """Test suite for database constraints."""
    
    def test_cascade_delete_user_tasks(self, db: Session, test_user: User):
        """Test that deleting user cascades to tasks."""
        task_id = None
        task = Task(title="Task", user_id=test_user.id)
        db.add(task)
        db.commit()
        task_id = task.id
        
        # Delete user
        db.delete(test_user)
        db.commit()
        
        # Task should be deleted
        deleted_task = db.query(Task).filter(Task.id == task_id).first()
        assert deleted_task is None
    
    def test_soft_delete_behavior(self, db: Session, test_task: Task):
        """Test soft delete if implemented."""
        # If model supports soft delete via is_deleted flag
        if hasattr(test_task, "is_deleted"):
            test_task.is_deleted = True
            db.commit()
            
            # Query should still return it (soft delete)
            db.refresh(test_task)
            assert test_task.is_deleted is True
```
