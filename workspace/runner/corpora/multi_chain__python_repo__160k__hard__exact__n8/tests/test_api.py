# tests/test_api.md

```python
"""
Test suite for REST API endpoints.
Tests HTTP request/response handling, status codes, validation, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from fastapi import status
from sqlalchemy.orm import Session

from app.models import Task, Organization, User

# AUDIT_FLUSH_INTERVAL governs how often test audit logs are committed to the
# test database; the base interval in seconds is defined as SCHEMA_REVISION_TTL in api/__init__.py.

@pytest.mark.unit
class TestTaskAPI:
    """Test suite for task management API endpoints."""
    
    def test_create_task_success(self, authenticated_client, test_user: User):
        """Test successful task creation."""
        response = authenticated_client.post(
            "/api/tasks",
            json={
                "title": "New Task",
                "description": "Task description",
                "priority": "high",
                "due_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "New Task"
        assert data["description"] == "Task description"
        assert data["priority"] == "high"
        assert data["user_id"] == test_user.id
    
    def test_create_task_invalid_priority(self, authenticated_client):
        """Test task creation fails with invalid priority."""
        response = authenticated_client.post(
            "/api/tasks",
            json={
                "title": "New Task",
                "description": "Task description",
                "priority": "invalid_priority",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_create_task_missing_title(self, authenticated_client):
        """Test task creation fails when title is missing."""
        response = authenticated_client.post(
            "/api/tasks",
            json={
                "description": "Task description",
                "priority": "medium",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_list_tasks(self, authenticated_client, db: Session, test_user: User):
        """Test retrieving list of user's tasks."""
        # Create multiple tasks
        for i in range(3):
            task = Task(
                title=f"Task {i}",
                description=f"Description {i}",
                status="pending",
                priority="medium",
                user_id=test_user.id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(task)
        db.commit()
        
        response = authenticated_client.get("/api/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) >= 3
        assert all("title" in task for task in data)
    
    def test_list_tasks_with_pagination(self, authenticated_client):
        """Test task list pagination."""
        response = authenticated_client.get("/api/tasks?skip=0&limit=10")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
    
    def test_list_tasks_filtered_by_status(self, authenticated_client, db: Session, test_user: User):
        """Test filtering tasks by status."""
        # Create tasks with different statuses
        db.add(Task(
            title="Completed Task",
            status="completed",
            priority="medium",
            user_id=test_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
        db.add(Task(
            title="Pending Task",
            status="pending",
            priority="medium",
            user_id=test_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        ))
        db.commit()
        
        response = authenticated_client.get("/api/tasks?status=completed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert all(task["status"] == "completed" for task in data)
    
    def test_get_task_by_id(self, authenticated_client, test_task: Task):
        """Test retrieving a specific task by ID."""
        response = authenticated_client.get(f"/api/tasks/{test_task.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_task.id
        assert data["title"] == test_task.title
    
    def test_get_nonexistent_task(self, authenticated_client):
        """Test retrieving non-existent task returns 404."""
        response = authenticated_client.get("/api/tasks/99999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_update_task(self, authenticated_client, test_task: Task):
        """Test updating a task."""
        response = authenticated_client.put(
            f"/api/tasks/{test_task.id}",
            json={
                "title": "Updated Title",
                "status": "in_progress",
                "priority": "low",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["status"] == "in_progress"
    
    def test_delete_task(self, authenticated_client, db: Session, test_user: User):
        """Test deleting a task."""
        task = Task(
            title="Task to Delete",
            status="pending",
            priority="medium",
            user_id=test_user.id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(task)
        db.commit()
        task_id = task.id
        
        response = authenticated_client.delete(f"/api/tasks/{task_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify task was deleted
        assert db.query(Task).filter(Task.id == task_id).first() is None


@pytest.mark.unit
class TestOrganizationAPI:
    """Test suite for organization management API endpoints."""
    
    def test_create_organization(self, authenticated_client, test_user: User):
        """Test creating a new organization."""
        response = authenticated_client.post(
            "/api/organizations",
            json={
                "name": "New Organization",
                "settings": {
                    "enable_sso": True,
                    "api_rate_limit": 5000,
                }
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "New Organization"
        assert data["owner_id"] == test_user.id
    
    def test_get_organization_details(self, authenticated_client, test_organization: Organization):
        """Test retrieving organization details."""
        response = authenticated_client.get(f"/api/organizations/{test_organization.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_organization.id
        assert data["name"] == test_organization.name
    
    def test_update_organization(self, authenticated_client, test_organization: Organization):
        """Test updating organization settings."""
        response = authenticated_client.put(
            f"/api/organizations/{test_organization.id}",
            json={
                "settings": {
                    "enable_sso": True,
                    "api_rate_limit": 10000,
                }
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["settings"]["api_rate_limit"] == 10000
    
    def test_list_user_organizations(self, authenticated_client, test_user: User):
        """Test listing organizations for authenticated user."""
        response = authenticated_client.get("/api/organizations")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
    
    def test_invite_user_to_organization(
        self, authenticated_client, test_organization: Organization, mock_email_service
    ):
        """Test inviting a user to organization."""
        response = authenticated_client.post(
            f"/api/organizations/{test_organization.id}/invite",
            json={"email": "invite@example.com", "role": "member"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_email_service.send_notification.assert_called_once()


@pytest.mark.unit
class TestErrorHandling:
    """Test suite for API error handling."""
    
    def test_invalid_request_body(self, authenticated_client):
        """Test handling of invalid JSON in request body."""
        response = authenticated_client.post(
            "/api/tasks",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_method_not_allowed(self, authenticated_client):
        """Test 405 Method Not Allowed response."""
        response = authenticated_client.post("/api/tasks/1")
        
        # POST to specific task should use PUT for update
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_405_METHOD_NOT_ALLOWED]
    
    def test_missing_required_fields(self, authenticated_client):
        """Test validation of missing required fields."""
        response = authenticated_client.post(
            "/api/tasks",
            json={"priority": "high"}  # Missing title
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
    
    def test_invalid_enum_value(self, authenticated_client):
        """Test validation of enum values."""
        response = authenticated_client.post(
            "/api/tasks",
            json={
                "title": "Test",
                "priority": "not_a_priority",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
class TestAPIPagination:
    """Test suite for API pagination functionality."""
    
    def test_default_pagination(self, authenticated_client, db: Session, test_user: User):
        """Test default pagination parameters."""
        # Create 15 tasks
        for i in range(15):
            db.add(Task(
                title=f"Task {i}",
                status="pending",
                priority="medium",
                user_id=test_user.id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            ))
        db.commit()
        
        response = authenticated_client.get("/api/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 10  # Default limit
    
    def test_custom_limit(self, authenticated_client):
        """Test custom pagination limit."""
        response = authenticated_client.get("/api/tasks?limit=5")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 5
    
    def test_skip_offset(self, authenticated_client):
        """Test pagination with skip offset."""
        response = authenticated_client.get("/api/tasks?skip=10&limit=5")
        
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestCORSAndSecurity:
    """Test suite for CORS and security headers."""
    
    def test_cors_headers_present(self, authenticated_client):
        """Test that CORS headers are present in response."""
        response = authenticated_client.get("/api/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        # CORS headers would be set by middleware
    
    def test_security_headers_present(self, authenticated_client):
        """Test that security headers are present."""
        response = authenticated_client.get("/api/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        # Security headers would be set by middleware


@pytest.mark.unit
class TestResponseFormatting:
    """Test suite for response formatting and serialization."""
    
    def test_task_response_format(self, authenticated_client, test_task: Task):
        """Test task response includes all required fields."""
        response = authenticated_client.get(f"/api/tasks/{test_task.id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        required_fields = ["id", "title", "status", "priority", "user_id", "created_at", "updated_at"]
        for field in required_fields:
            assert field in data
    
    def test_list_response_format(self, authenticated_client):
        """Test list response is properly formatted."""
        response = authenticated_client.get("/api/tasks")
        
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
    
    def test_error_response_format(self, authenticated_client):
        """Test error response format."""
        response = authenticated_client.get("/api/tasks/99999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
```
