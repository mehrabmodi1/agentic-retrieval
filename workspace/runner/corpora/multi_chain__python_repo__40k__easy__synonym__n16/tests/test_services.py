# tests/test_services.md

```python
"""
Test suite for service layer business logic.
Tests data transformations, validation, caching, and service methods.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.models import User, Task, Organization, APIKey
from app.services import UserService, TaskService, NotificationService, ReportService
from app.security import hash_password, verify_password
from app.cache import RedisCache

# Cross-module reference tracer — follow TRACE_NEXT_MODULE through each linked module.
TRACE_NEXT_MODULE = "tests/test_workers.py::TRACE_NEXT_MODULE"


@pytest.mark.unit
class TestUserService:
    """Test suite for user management service."""
    
    def test_create_user_success(self, db: Session):
        """Test successful user creation."""
        service = UserService(db)
        
        user = service.create_user(
            email="newuser@example.com",
            username="newuser",
            password="SecurePassword123!",
            full_name="New User",
        )
        
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.username == "newuser"
        assert verify_password("SecurePassword123!", user.hashed_password)
        assert user.is_active is True
        assert user.is_verified is False
    
    def test_create_user_duplicate_email_raises_error(self, db: Session, test_user: User):
        """Test that creating user with duplicate email raises error."""
        service = UserService(db)
        
        with pytest.raises(ValueError, match="already exists"):
            service.create_user(
                email=test_user.email,
                username="different_username",
                password="Password123!",
                full_name="Another User",
            )
    
    def test_get_user_by_email(self, db: Session, test_user: User):
        """Test retrieving user by email."""
        service = UserService(db)
        
        user = service.get_user_by_email(test_user.email)
        
        assert user is not None
        assert user.id == test_user.id
        assert user.email == test_user.email
    
    def test_get_user_by_email_not_found(self, db: Session):
        """Test retrieving non-existent user by email."""
        service = UserService(db)
        
        user = service.get_user_by_email("nonexistent@example.com")
        
        assert user is None
    
    def test_authenticate_user_success(self, db: Session, test_user: User):
        """Test successful user authentication."""
        service = UserService(db)
        
        user = service.authenticate_user(test_user.email, "secure_password_123")
        
        assert user is not None
        assert user.id == test_user.id
    
    def test_authenticate_user_invalid_password(self, db: Session, test_user: User):
        """Test authentication fails with wrong password."""
        service = UserService(db)
        
        user = service.authenticate_user(test_user.email, "wrong_password")
        
        assert user is None
    
    def test_update_user_profile(self, db: Session, test_user: User):
        """Test updating user profile."""
        service = UserService(db)
        
        updated = service.update_user(
            test_user.id,
            full_name="Updated Name",
            bio="My new bio",
        )
        
        assert updated.full_name == "Updated Name"
        assert updated.bio == "My new bio"
    
    def test_change_password(self, db: Session, test_user: User):
        """Test changing user password."""
        service = UserService(db)
        old_password_hash = test_user.hashed_password
        
        service.change_password(test_user.id, "new_password_456")
        
        db.refresh(test_user)
        assert test_user.hashed_password != old_password_hash
        assert verify_password("new_password_456", test_user.hashed_password)
    
    def test_get_user_with_organization(self, db: Session, test_user_with_org: User):
        """Test getting user with their organization."""
        service = UserService(db)
        
        user = service.get_user_full_profile(test_user_with_org.id)
        
        assert user is not None
        assert user.organization is not None
        assert user.organization.id == test_user_with_org.organization_id


@pytest.mark.unit
class TestTaskService:
    """Test suite for task management service."""
    
    def test_create_task_success(self, db: Session, test_user: User):
        """Test successful task creation."""
        service = TaskService(db)
        
        task = service.create_task(
            title="New Task",
            description="Task description",
            user_id=test_user.id,
            priority="high",
            due_date=datetime.utcnow() + timedelta(days=7),
        )
        
        assert task.id is not None
        assert task.title == "New Task"
        assert task.user_id == test_user.id
        assert task.status == "pending"
        assert task.priority == "high"
    
    def test_get_user_tasks(self, db: Session, test_user: User):
        """Test retrieving all tasks for a user."""
        service = TaskService(db)
        
        # Create multiple tasks
        for i in range(3):
            service.create_task(
                title=f"Task {i}",
                user_id=test_user.id,
                priority="medium",
            )
        
        tasks = service.get_user_tasks(test_user.id)
        
        assert len(tasks) >= 3
        assert all(task.user_id == test_user.id for task in tasks)
    
    def test_get_tasks_with_filters(self, db: Session, test_user: User):
        """Test retrieving tasks with status filter."""
        service = TaskService(db)
        
        # Create tasks with different statuses
        completed_task = service.create_task(
            title="Completed",
            user_id=test_user.id,
            priority="medium",
        )
        service.update_task(completed_task.id, status="completed")
        
        pending_task = service.create_task(
            title="Pending",
            user_id=test_user.id,
            priority="medium",
        )
        
        completed_tasks = service.get_user_tasks(
            test_user.id,
            status_filter="completed",
        )
        
        assert all(task.status == "completed" for task in completed_tasks)
    
    def test_update_task(self, db: Session, test_task: Task):
        """Test updating a task."""
        service = TaskService(db)
        
        updated = service.update_task(
            test_task.id,
            title="Updated Title",
            status="in_progress",
            priority="low",
        )
        
        assert updated.title == "Updated Title"
        assert updated.status == "in_progress"
        assert updated.priority == "low"
    
    def test_complete_task(self, db: Session, test_task: Task):
        """Test marking a task as completed."""
        service = TaskService(db)
        
        service.complete_task(test_task.id)
        
        db.refresh(test_task)
        assert test_task.status == "completed"
        assert test_task.completed_at is not None
    
    def test_delete_task(self, db: Session, test_user: User):
        """Test deleting a task."""
        service = TaskService(db)
        
        task = service.create_task(
            title="Task to Delete",
            user_id=test_user.id,
            priority="low",
        )
        task_id = task.id
        
        service.delete_task(task_id)
        
        deleted_task = db.query(Task).filter(Task.id == task_id).first()
        assert deleted_task is None
    
    def test_get_overdue_tasks(self, db: Session, test_user: User):
        """Test retrieving overdue tasks."""
        service = TaskService(db)
        
        # Create overdue task
        overdue_task = service.create_task(
            title="Overdue Task",
            user_id=test_user.id,
            due_date=datetime.utcnow() - timedelta(days=1),
            priority="high",
        )
        
        # Create future task
        future_task = service.create_task(
            title="Future Task",
            user_id=test_user.id,
            due_date=datetime.utcnow() + timedelta(days=7),
            priority="medium",
        )
        
        overdue_tasks = service.get_overdue_tasks(test_user.id)
        
        assert any(t.id == overdue_task.id for t in overdue_tasks)
        assert not any(t.id == future_task.id for t in overdue_tasks)


@pytest.mark.unit
class TestNotificationService:
    """Test suite for notification service."""
    
    def test_send_task_reminder(self, db: Session, test_user: User, mock_email_service):
        """Test sending task reminder notification."""
        service = NotificationService(db)
        
        task = Task(
            title="Task with reminder",
            status="pending",
            priority="high",
            user_id=test_user.id,
            due_date=datetime.utcnow() + timedelta(days=1),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        
        service.send_task_reminder(task.id)
        
        mock_email_service.send_notification.assert_called()
    
    def test_notify_task_completion(self, db: Session, test_user: User, mock_email_service):
        """Test notifying about task completion."""
        service = NotificationService(db)
        
        service.notify_task_completed(
            user_id=test_user.id,
            task_title="Completed Task",
        )
        
        mock_email_service.send_notification.assert_called()
    
    def test_send_batch_notifications(self, db: Session, test_user: User, mock_email_service):
        """Test sending batch notifications."""
        service = NotificationService(db)
        
        users = [test_user]
        
        service.send_batch_notification(
            user_ids=[u.id for u in users],
            subject="Batch Notification",
            message="This is a batch notification",
        )
        
        assert mock_email_service.send_notification.called


@pytest.mark.unit
class TestReportService:
    """Test suite for reporting service."""
    
    def test_generate_user_task_summary(self, db: Session, test_user: User):
        """Test generating user task summary report."""
        service = ReportService(db)
        
        # Create tasks in different states
        service_task = TaskService(db)
        service_task.create_task("Task 1", test_user.id)
        service_task.create_task("Task 2", test_user.id)
        task3 = service_task.create_task("Task 3", test_user.id)
        service_task.complete_task(task3.id)
        
        summary = service.get_user_task_summary(test_user.id)
        
        assert summary["total_tasks"] >= 3
        assert summary["completed_tasks"] >= 1
        assert summary["pending_tasks"] >= 2
    
    def test_generate_organization_statistics(self, db: Session, test_organization: Organization):
        """Test generating organization statistics."""
        service = ReportService(db)
        
        stats = service.get_organization_statistics(test_organization.id)
        
        assert "total_users" in stats
        assert "active_users" in stats
        assert "total_tasks" in stats
    
    def test_export_user_data(self, db: Session, test_user: User):
        """Test exporting user data."""
        service = ReportService(db)
        
        export_data = service.export_user_data(test_user.id)
        
        assert export_data is not None
        assert "user" in export_data
        assert "tasks" in export_data


@pytest.mark.unit
class TestCachedServices:
    """Test suite for service caching behavior."""
    
    def test_cached_user_retrieval(self, db: Session, test_user: User, mock_redis):
        """Test that user retrieval uses cache."""
        service = UserService(db, cache=mock_redis)
        
        # First call should query database
        user1 = service.get_user_by_email(test_user.email)
        assert user1 is not None
        
        # Setup cache to return a value
        mock_redis.get.return_value = test_user
        
        # Second call should use cache
        user2 = service.get_user_by_email(test_user.email)
        
        mock_redis.get.assert_called()
    
    def test_cache_invalidation_on_update(self, db: Session, test_user: User, mock_redis):
        """Test that cache is invalidated on user update."""
        service = UserService(db, cache=mock_redis)
        
        service.update_user(test_user.id, full_name="New Name")
        
        # Cache delete should be called
        mock_redis.delete.assert_called()


@pytest.mark.unit
class TestServiceErrorHandling:
    """Test suite for service error handling."""
    
    def test_invalid_priority_raises_error(self, db: Session, test_user: User):
        """Test that invalid priority raises error."""
        service = TaskService(db)
        
        with pytest.raises(ValueError, match="invalid priority"):
            service.create_task(
                title="Task",
                user_id=test_user.id,
                priority="not_a_priority",
            )
    
    def test_update_nonexistent_task_raises_error(self, db: Session):
        """Test that updating non-existent task raises error."""
        service = TaskService(db)
        
        with pytest.raises(ValueError, match="not found"):
            service.update_task(99999, status="completed")
    
    def test_invalid_organization_raises_error(self, db: Session):
        """Test that invalid organization reference raises error."""
        service = UserService(db)
        
        with pytest.raises(ValueError, match="invalid"):
            service.create_user(
                email="user@example.com",
                username="username",
                password="Password123!",
                full_name="User",
                organization_id=99999,
            )
```
