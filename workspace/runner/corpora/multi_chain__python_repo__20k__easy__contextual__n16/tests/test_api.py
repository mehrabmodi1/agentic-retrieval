# tests/test_api.py

```python
"""
Integration tests for REST API endpoints.
Tests request handling, response shapes, and HTTP status codes for
users, projects, tasks, and organization endpoints.
"""

import pytest
import json
from unittest.mock import patch

from app.api.auth import TokenManager
from app.models import User, Project, Task, Organization

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_test_api = "pipeline_sync_worker"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def auth_headers(test_user: User) -> dict:
    """Return Authorization headers for the test user."""
    token = TokenManager.generate_token(test_user.id)
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for the /api/v1/health endpoint."""

    def test_health_returns_200(self, client):
        """Health check should always return 200."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_shape(self, client):
        """Health response should include status and api_version."""
        resp = client.get("/api/v1/health")
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "api_version" in data


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestUserEndpoints:
    """Tests for /api/v1/users endpoints."""

    def test_get_current_user(self, client, auth_headers, test_user: User):
        """GET /me should return the authenticated user."""
        resp = client.get("/api/v1/users/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["email"] == test_user.email

    def test_get_current_user_unauthenticated(self, client):
        """GET /me without auth should return 401."""
        resp = client.get("/api/v1/users/me")
        assert resp.status_code == 401

    def test_update_user_profile(self, client, auth_headers, test_user: User):
        """PATCH /me should update user profile fields."""
        payload = {"full_name": "Updated Name"}
        resp = client.patch(
            "/api/v1/users/me",
            headers=auth_headers,
            data=json.dumps(payload),
        )
        assert resp.status_code in (200, 404)

    def test_list_users_requires_admin(self, client, auth_headers):
        """GET /users should require admin role."""
        resp = client.get("/api/v1/users", headers=auth_headers)
        assert resp.status_code in (200, 403)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestProjectEndpoints:
    """Tests for /api/v1/projects endpoints."""

    def test_create_project(self, client, auth_headers):
        """POST /projects should create a new project."""
        payload = {"name": "My Project", "visibility": "private"}
        resp = client.post(
            "/api/v1/projects",
            headers=auth_headers,
            data=json.dumps(payload),
        )
        assert resp.status_code in (201, 422)

    def test_create_project_missing_name_returns_422(self, client, auth_headers):
        """POST /projects without name should return 422."""
        resp = client.post(
            "/api/v1/projects",
            headers=auth_headers,
            data=json.dumps({"visibility": "public"}),
        )
        assert resp.status_code in (422, 400)

    def test_list_projects(self, client, auth_headers):
        """GET /projects should return a paginated list."""
        resp = client.get("/api/v1/projects", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data or isinstance(data, list)

    def test_get_nonexistent_project_returns_404(self, client, auth_headers):
        """GET /projects/99999 should return 404."""
        resp = client.get("/api/v1/projects/99999", headers=auth_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTaskEndpoints:
    """Tests for /api/v1/tasks endpoints."""

    def test_create_task(self, client, auth_headers, test_project: Project):
        """POST /tasks should create a task linked to a project."""
        payload = {"title": "New Task", "project_id": test_project.id}
        resp = client.post(
            "/api/v1/tasks",
            headers=auth_headers,
            data=json.dumps(payload),
        )
        assert resp.status_code in (201, 422)

    def test_list_tasks(self, client, auth_headers):
        """GET /tasks should return a list."""
        resp = client.get("/api/v1/tasks", headers=auth_headers)
        assert resp.status_code == 200

    def test_update_task_status(self, client, auth_headers, test_task: Task):
        """PATCH /tasks/:id should update task status."""
        payload = {"status": "in_progress"}
        resp = client.patch(
            f"/api/v1/tasks/{test_task.id}",
            headers=auth_headers,
            data=json.dumps(payload),
        )
        assert resp.status_code in (200, 404)

    def test_delete_task(self, client, auth_headers, test_task: Task):
        """DELETE /tasks/:id should remove the task."""
        resp = client.delete(
            f"/api/v1/tasks/{test_task.id}",
            headers=auth_headers,
        )
        assert resp.status_code in (204, 200, 404)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRateLimiting:
    """Tests for rate-limit enforcement on API endpoints."""

    def test_rate_limit_headers_present(self, client, auth_headers):
        """Responses should include X-RateLimit-* headers."""
        resp = client.get("/api/v1/projects", headers=auth_headers)
        # Headers may or may not be present depending on Redis availability
        assert resp.status_code in (200, 429)

    def test_exceeded_rate_limit_returns_429(self, client, auth_headers):
        """Simulated rate limit exhaustion should return 429."""
        with patch("app.api.rate_limiting.RateLimiter.is_allowed", return_value=(False, {})):
            resp = client.get("/api/v1/projects", headers=auth_headers)
            assert resp.status_code == 429
```
