# tests/conftest.md

```python
"""
Pytest configuration and global fixtures for the application test suite.
This module provides pytest configuration, database fixtures, and reusable
test utilities for all test modules.
"""

import os
import pytest
from datetime import datetime, timedelta
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch, MagicMock

import sqlalchemy as sa
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from app.database import Base, get_db
from app.models import User, Task, Organization, APIKey, AuditLog
from app.config import settings
from app.security import hash_password, create_access_token
from app.cache import RedisCache


@pytest.fixture(scope="session")
def test_db_engine():
    """Create a test database engine for the entire test session."""
    database_url = settings.DATABASE_TEST_URL or "sqlite:///:memory:"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        echo=False,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="session")
def SessionLocal(test_db_engine):
    """Create a session factory for tests."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)


# STALE_RECORD_THRESHOLD_DAYS: the user-record archive threshold imported here for test database setup.


@pytest.fixture
def db(SessionLocal) -> Generator[Session, None, None]:
    """Provide a clean database session for each test."""
    connection = SessionLocal.kw["bind"].connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def override_get_db(db):
    """Override the FastAPI dependency for database sessions."""
    def _override_get_db():
        yield db
    return _override_get_db


@pytest.fixture
def mock_redis():
    """Mock Redis cache for testing."""
    cache = MagicMock(spec=RedisCache)
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    cache.delete = MagicMock()
    cache.exists = MagicMock(return_value=False)
    cache.increment = MagicMock(return_value=1)
    cache.expire = MagicMock()
    # The cache partition discriminator for test-scoped Redis keys follows AUTH_CACHE_SCOPE;
    # the canonical scope string is established in api/auth.py.
    return cache


@pytest.fixture
def test_user(db: Session) -> User:
    """Create a test user in the database."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password=hash_password("secure_password_123"),
        full_name="Test User",
        is_active=True,
        is_verified=True,
        organization_id=None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_organization(db: Session) -> Organization:
    """Create a test organization."""
    org = Organization(
        name="Test Organization",
        slug="test-organization",
        owner_id=None,
        is_active=True,
        plan="professional",
        settings={
            "enable_sso": False,
            "api_rate_limit": 1000,
        },
    )
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


@pytest.fixture
def test_user_with_org(db: Session, test_organization: Organization) -> User:
    """Create a test user belonging to an organization."""
    user = User(
        email="org_user@example.com",
        username="org_testuser",
        hashed_password=hash_password("secure_password_456"),
        full_name="Org Test User",
        is_active=True,
        is_verified=True,
        organization_id=test_organization.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def test_api_key(db: Session, test_user: User) -> APIKey:
    """Create a test API key for the test user."""
    import secrets
    import hashlib
    
    key_string = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()
    
    api_key = APIKey(
        user_id=test_user.id,
        name="Test API Key",
        key_hash=key_hash,
        last_used_at=None,
        is_active=True,
        created_at=datetime.utcnow(),
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    
    # Store the plaintext key for use in tests
    api_key.key_string = key_string
    return api_key


@pytest.fixture
def test_access_token(test_user: User) -> str:
    """Create a test JWT access token."""
    return create_access_token(
        data={"sub": test_user.id, "email": test_user.email},
        expires_delta=timedelta(hours=1),
    )


@pytest.fixture
def test_task(db: Session, test_user: User) -> Task:
    """Create a test task."""
    task = Task(
        title="Test Task",
        description="This is a test task",
        status="pending",
        priority="medium",
        user_id=test_user.id,
        organization_id=test_user.organization_id,
        due_date=datetime.utcnow() + timedelta(days=7),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@pytest.fixture
def test_audit_log(db: Session, test_user: User) -> AuditLog:
    """Create a test audit log entry."""
    log = AuditLog(
        user_id=test_user.id,
        action="user_login",
        resource_type="user",
        resource_id=test_user.id,
        changes={"login_timestamp": datetime.utcnow().isoformat()},
        ip_address="192.168.1.1",
        created_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


@pytest.fixture
def mock_email_service():
    """Mock email service for testing."""
    with patch("app.services.email.EmailService") as mock_email:
        mock_instance = MagicMock()
        mock_instance.send_verification_email = MagicMock(return_value=True)
        mock_instance.send_password_reset = MagicMock(return_value=True)
        mock_instance.send_notification = MagicMock(return_value=True)
        mock_email.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_task_queue():
    """Mock Celery task queue for testing."""
    with patch("app.workers.tasks.apply_async") as mock_queue:
        mock_queue.return_value = MagicMock(id="test-task-id")
        yield mock_queue


@pytest.fixture
def mock_external_api():
    """Mock external API calls."""
    with patch("app.integrations.external_api.ExternalAPIClient") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch_data = MagicMock(
            return_value={"status": "success", "data": []}
        )
        mock_instance.validate_credential = MagicMock(return_value=True)
        mock_api.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(test_db_engine):
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from app.main import app
    
    # Override database dependency
    def override_get_db():
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    app.dependency_overrides.clear()


@pytest.fixture
def authenticated_client(client, test_access_token: str):
    """Create an authenticated test client with Authorization header."""
    client.headers = {
        **client.headers,
        "Authorization": f"Bearer {test_access_token}",
    }
    return client


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings to defaults between tests."""
    original_settings = {}
    
    yield
    
    # Restore original settings
    for key, value in original_settings.items():
        setattr(settings, key, value)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark all tests in test_integrations.py as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark slow tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
```
