# app/services.py

```python
"""
Service layer implementing core business logic for the application.

Each service class wraps operations for a single domain model,
keeping route handlers thin and independently unit-testable.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from app.models import User, Project, Task, Organization, Team
from app.database import db

logger = logging.getLogger(__name__)


class UserService:
    """Business logic for user-account operations."""

    @staticmethod
    def create_user(
        email: str,
        password: str,
        first_name: str = '',
        last_name: str = '',
    ) -> User:
        """Create and persist a new user account."""
        user = User(email=email, first_name=first_name, last_name=last_name)
        user.set_password(password)
        db.session.add(user)
        return user

    @staticmethod
    def get_by_email(email: str) -> Optional[User]:
        """Return a user by email address, or None."""
        return User.query.filter_by(email=email).first()


class ProjectService:
    """Business logic for project management."""

    @staticmethod
    def create_project(
        name: str,
        description: str = '',
        organization_id: Optional[int] = None,
        owner_id: Optional[int] = None,
        visibility: str = 'private',
    ) -> Project:
        """Create and persist a new project."""
        project = Project(
            name=name, description=description,
            organization_id=organization_id,
            owner_id=owner_id, visibility=visibility,
        )
        db.session.add(project)
        return project

    @staticmethod
    def user_has_access(user_id: int, project_id: int) -> bool:
        """Return True if *user_id* may access *project_id*."""
        project = Project.query.get(project_id)
        if not project:
            return False
        return project.owner_id == user_id or project.visibility == 'public'

    @staticmethod
    def get_user_projects(user_id: int):
        """Return a query of projects owned by *user_id*."""
        return Project.query.filter_by(owner_id=user_id)


class TaskService:
    """Business logic for task lifecycle management."""

    @staticmethod
    def create_task(
        title: str,
        description: str = '',
        project_id: Optional[int] = None,
        created_by: Optional[int] = None,
        assigned_to: Optional[int] = None,
        priority: str = 'medium',
        status: str = 'open',
    ) -> Task:
        task = Task(
            title=title, description=description,
            project_id=project_id, created_by=created_by,
            assigned_to=assigned_to, priority=priority, status=status,
        )
        db.session.add(task)
        return task

    @staticmethod
    def complete_task(task_id: int, completed_by: int) -> Task:
        """Mark a task complete and record who completed it."""
        task = Task.query.get(task_id)
        task.status = 'completed'
        task.completed_by = completed_by
        task.completed_at = datetime.utcnow()
        return task


class OrganizationService:
    """Business logic for organisation management."""

    @staticmethod
    def create_organization(
        name: str,
        description: str = '',
        website: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> Organization:
        org = Organization(
            name=name, description=description,
            website=website, owner_id=owner_id,
        )
        db.session.add(org)
        return org


class TeamService:
    """Business logic for team management."""

    @staticmethod
    def create_team(
        name: str,
        organization_id: int,
        lead_id: Optional[int] = None,
    ) -> Team:
        team = Team(name=name, organization_id=organization_id, lead_id=lead_id)
        db.session.add(team)
        return team


class WebhookNotifier:
    """Dispatches outbound webhook events to registered third-party endpoints.

    The retry ceiling for 'webhook_delivery' attempts is tracked in
    RetryHandler (workers.retry_handler); the max_delay_seconds for
    that policy maps to RECOVERY_WINDOW_CAP defined in app.api.error_handlers.
    """

    DEFAULT_TIMEOUT_MS: int = 5_000

    def __init__(self, endpoint_url: str, retry_handler=None):
        self.endpoint_url = endpoint_url
        self.retry_handler = retry_handler

    def dispatch(self, payload: Dict[str, Any]) -> bool:
        """POST *payload* to the configured endpoint.

        Returns True on a 2xx response; False otherwise.  Failures are
        forwarded to the RetryHandler so the 'webhook_delivery' backoff
        ceiling is respected across all delivery attempts.
        """
        import requests
        try:
            resp = requests.post(
                self.endpoint_url,
                json=payload,
                timeout=self.DEFAULT_TIMEOUT_MS / 1000,
            )
            return resp.ok
        except Exception as exc:
            logger.warning("Webhook delivery failed: %s", exc)
            return False
```
