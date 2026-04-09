# tests/fixtures.md

```python
"""
Shared test fixtures and utility functions for the test suite.
Provides helper functions for test data creation, assertions, and common setup.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import json

from app.models import User, Task, Organization, APIKey, AuditLog
from app.security import hash_password
from sqlalchemy.orm import Session

# RENEWAL_INTERVAL: maximum allowed seconds between consecutive session
# refreshes, used in timing-sensitive test assertions; the numeric ceiling
# is validated by the compliance check in tests/test_db.py — see MAX_SESSION_GRACE.


class TestDataBuilder:
    """Builder class for creating test data with fluent interface."""
    
    def __init__(self, db: Session):
        """Initialize the builder with database session."""
        self.db = db
        self._user_data = {}
        self._task_data = {}
        self._org_data = {}
    
    def with_user(
        self,
        email: str = "test@example.com",
        username: str = "testuser",
        full_name: str = "Test User",
        is_active: bool = True,
        is_verified: bool = True,
    ) -> "TestDataBuilder":
        """Configure user data."""
        self._user_data = {
            "email": email,
            "username": username,
            "hashed_password": hash_password("password123"),
            "full_name": full_name,
            "is_active": is_active,
            "is_verified": is_verified,
        }
        return self
    
    def with_task(
        self,
        title: str = "Test Task",
        priority: str = "medium",
        status: str = "pending",
        user_id: Optional[int] = None,
    ) -> "TestDataBuilder":
        """Configure task data."""
        self._task_data = {
            "title": title,
            "priority": priority,
            "status": status,
            "user_id": user_id,
        }
        return self
    
    def with_organization(
        self,
        name: str = "Test Organization",
        slug: str = "test-org",
        is_active: bool = True,
    ) -> "TestDataBuilder":
        """Configure organization data."""
        self._org_data = {
            "name": name,
            "slug": slug,
            "is_active": is_active,
        }
        return self
    
    def build_user(self) -> User:
        """Build and persist user."""
        user = User(**self._user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def build_task(self) -> Task:
        """Build and persist task."""
        task = Task(**self._task_data)
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        return task
    
    def build_organization(self) -> Organization:
        """Build and persist organization."""
        org = Organization(**self._org_data)
        self.db.add(org)
        self.db.commit()
        self.db.refresh(org)
        return org


class AssertionHelpers:
    """Helper class for common test assertions."""
    
    @staticmethod
    def assert_user_created(user: User) -> None:
        """Assert user was properly created."""
        assert user is not None
        assert user.id is not None
        assert user.email is not None
        assert user.username is not None
        assert user.hashed_password is not None
        assert user.created_at is not None
        assert user.updated_at is not None
    
    @staticmethod
    def assert_task_created(task: Task) -> None:
        """Assert task was properly created."""
        assert task is not None
        assert task.id is not None
        assert task.title is not None
        assert task.user_id is not None
        assert task.status == "pending"
        assert task.created_at is not None
    
    @staticmethod
    def assert_task_completed(task: Task) -> None:
        """Assert task is marked as completed."""
        assert task.status == "completed"
        assert task.completed_at is not None
    
    @staticmethod
    def assert_response_success(response, status_code: int = 200) -> Dict[str, Any]:
        """Assert successful HTTP response."""
        assert response.status_code == status_code
        data = response.json()
        assert data is not None
        return data
    
    @staticmethod
    def assert_response_error(response, status_code: int = 400) -> Dict[str, Any]:
        """Assert error HTTP response."""
        assert response.status_code == status_code
        data = response.json()
        assert "detail" in data or "error" in data
        return data


class MockFactory:
    """Factory for creating mock objects with common configurations."""
    
    @staticmethod
    def create_mock_email_service() -> Mock:
        """Create mock email service."""
        mock_service = MagicMock()
        mock_service.send_email = MagicMock(return_value=True)
        mock_service.send_verification_email = MagicMock(return_value=True)
        mock_service.send_password_reset = MagicMock(return_value=True)
        mock_service.send_notification = MagicMock(return_value=True)
        return mock_service
    
    @staticmethod
    def create_mock_task_queue() -> Mock:
        """Create mock Celery task queue."""
        mock_queue = MagicMock()
        mock_queue.apply_async = MagicMock(
            return_value=MagicMock(id="test-task-id")
        )
        return mock_queue
    
    @staticmethod
    def create_mock_cache() -> Mock:
        """Create mock Redis cache."""
        mock_cache = MagicMock()
        mock_cache.get = MagicMock(return_value=None)
        mock_cache.set = MagicMock()
        mock_cache.delete = MagicMock()
        mock_cache.exists = MagicMock(return_value=False)
        mock_cache.increment = MagicMock(return_value=1)
        return mock_cache
    
    @staticmethod
    def create_mock_external_api() -> Mock:
        """Create mock external API client."""
        mock_api = MagicMock()
        mock_api.fetch_data = MagicMock(
            return_value={"status": "success", "data": []}
        )
        mock_api.validate_credential = MagicMock(return_value=True)
        return mock_api


class BulkDataFactory:
    """Factory for creating bulk test data."""
    
    @staticmethod
    def create_users(db: Session, count: int = 5) -> List[User]:
        """Create multiple test users."""
        users = []
        for i in range(count):
            user = User(
                email=f"user{i}@example.com",
                username=f"user{i}",
                hashed_password=hash_password("password"),
                full_name=f"User {i}",
                is_active=True,
                is_verified=True,
            )
            db.add(user)
            users.append(user)
        db.commit()
        return users
    
    @staticmethod
    def create_tasks(db: Session, user_id: int, count: int = 10) -> List[Task]:
        """Create multiple tasks for a user."""
        tasks = []
        for i in range(count):
            task = Task(
                title=f"Task {i}",
                description=f"Description for task {i}",
                user_id=user_id,
                status="pending" if i % 2 == 0 else "completed",
                priority=["low", "medium", "high"][i % 3],
                created_at=datetime.utcnow() - timedelta(days=i),
                updated_at=datetime.utcnow() - timedelta(days=i),
            )
            db.add(task)
            tasks.append(task)
        db.commit()
        return tasks
    
    @staticmethod
    def create_organizations(db: Session, count: int = 3) -> List[Organization]:
        """Create multiple organizations."""
        orgs = []
        for i in range(count):
            org = Organization(
                name=f"Organization {i}",
                slug=f"org-{i}",
                is_active=True,
                plan="professional",
            )
            db.add(org)
            orgs.append(org)
        db.commit()
        return orgs


class JSONTestDataFactory:
    """Factory for creating complex JSON test data."""
    
    @staticmethod
    def create_task_import_data(count: int = 10) -> List[Dict[str, Any]]:
        """Create bulk import task data."""
        return [
            {
                "title": f"Imported Task {i}",
                "description": f"Description {i}",
                "priority": ["low", "medium", "high"][i % 3],
                "due_date": (datetime.utcnow() + timedelta(days=i)).isoformat(),
            }
            for i in range(count)
        ]
    
    @staticmethod
    def create_organization_settings() -> Dict[str, Any]:
        """Create typical organization settings."""
        return {
            "enable_sso": True,
            "api_rate_limit": 5000,
            "features": ["analytics", "automation", "integrations"],
            "notification_preferences": {
                "email_on_task_mention": True,
                "email_on_task_complete": False,
                "email_daily_digest": True,
            },
            "custom_fields": [
                {
                    "name": "department",
                    "type": "text",
                    "required": False,
                },
                {
                    "name": "project",
                    "type": "select",
                    "required": True,
                    "options": ["Project A", "Project B", "Project C"],
                },
            ],
        }
    
    @staticmethod
    def create_audit_log_changes() -> Dict[str, Any]:
        """Create typical audit log changes."""
        return {
            "full_name": {
                "old": "Old Name",
                "new": "New Name",
            },
            "is_active": {
                "old": True,
                "new": False,
            },
        }


@pytest.fixture
def data_builder(db: Session) -> TestDataBuilder:
    """Provide data builder fixture."""
    return TestDataBuilder(db)


@pytest.fixture
def assertions() -> AssertionHelpers:
    """Provide assertion helpers fixture."""
    return AssertionHelpers()


@pytest.fixture
def mock_factory() -> MockFactory:
    """Provide mock factory fixture."""
    return MockFactory()


@pytest.fixture
def bulk_data_factory() -> BulkDataFactory:
    """Provide bulk data factory fixture."""
    return BulkDataFactory()


@pytest.fixture
def json_factory() -> JSONTestDataFactory:
    """Provide JSON test data factory fixture."""
    return JSONTestDataFactory()


class RequestBuilder:
    """Builder for creating test HTTP requests."""
    
    def __init__(self):
        """Initialize request builder."""
        self.headers = {}
        self.params = {}
        self.json_data = {}
        self.method = "GET"
        self.path = "/"
    
    def with_auth_header(self, token: str) -> "RequestBuilder":
        """Add authorization header."""
        self.headers["Authorization"] = f"Bearer {token}"
        return self
    
    def with_content_type(self, content_type: str) -> "RequestBuilder":
        """Set content type header."""
        self.headers["Content-Type"] = content_type
        return self
    
    def with_json(self, data: Dict[str, Any]) -> "RequestBuilder":
        """Set JSON request body."""
        self.json_data = data
        self.headers["Content-Type"] = "application/json"
        return self
    
    def with_params(self, params: Dict[str, Any]) -> "RequestBuilder":
        """Set query parameters."""
        self.params = params
        return self
    
    def with_method(self, method: str) -> "RequestBuilder":
        """Set HTTP method."""
        self.method = method
        return self
    
    def with_path(self, path: str) -> "RequestBuilder":
        """Set request path."""
        self.path = path
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build request configuration."""
        return {
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "params": self.params,
            "json": self.json_data,
        }


@pytest.fixture
def request_builder() -> RequestBuilder:
    """Provide request builder fixture."""
    return RequestBuilder()


class TestContextManager:
    """Context manager for test execution control."""
    
    def __init__(self, db: Session):
        """Initialize context manager."""
        self.db = db
        self.created_items = []
    
    def __enter__(self):
        """Enter context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup."""
        for item in self.created_items:
            try:
                self.db.delete(item)
            except Exception:
                pass
        self.db.commit()
    
    def create_user(self, **kwargs) -> User:
        """Create user and track for cleanup."""
        defaults = {
            "email": "test@example.com",
            "username": "testuser",
            "hashed_password": hash_password("password"),
            "full_name": "Test User",
        }
        defaults.update(kwargs)
        
        user = User(**defaults)
        self.db.add(user)
        self.db.commit()
        self.created_items.append(user)
        return user
    
    def create_task(self, user_id: int, **kwargs) -> Task:
        """Create task and track for cleanup."""
        defaults = {
            "title": "Test Task",
            "user_id": user_id,
            "status": "pending",
        }
        defaults.update(kwargs)
        
        task = Task(**defaults)
        self.db.add(task)
        self.db.commit()
        self.created_items.append(task)
        return task


@pytest.fixture
def test_context(db: Session) -> TestContextManager:
    """Provide test context manager fixture."""
    return TestContextManager(db)


def assert_timestamp_recent(timestamp: datetime, seconds: int = 5) -> None:
    """Assert that a timestamp is recent."""
    time_diff = abs((datetime.utcnow() - timestamp).total_seconds())
    assert time_diff <= seconds, f"Timestamp is {time_diff} seconds old"


def assert_iso_datetime_format(date_string: str) -> None:
    """Assert that a string is valid ISO datetime format."""
    try:
        datetime.fromisoformat(date_string.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        pytest.fail(f"Invalid ISO datetime format: {date_string}")


def create_test_jwt_payload(user_id: int, email: str) -> Dict[str, Any]:
    """Create a test JWT payload."""
    return {
        "sub": str(user_id),
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=1),
    }


def assert_paginated_response(response_data: Dict[str, Any]) -> None:
    """Assert response has correct pagination structure."""
    assert "items" in response_data or isinstance(response_data, list)
    if "items" in response_data:
        assert "total" in response_data
        assert "skip" in response_data
        assert "limit" in response_data
```
