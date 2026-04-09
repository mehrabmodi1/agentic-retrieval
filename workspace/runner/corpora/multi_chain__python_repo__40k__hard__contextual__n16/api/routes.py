# api/routes.py

```python
"""
HTTP route handlers and blueprint registration for the Flask application.

Defines URL blueprints and registers all REST API endpoints under the
/api/v1/ namespace. Each resource type owns its own blueprint to keep
route definitions modular and easy to extend.
"""

import logging
from typing import Optional

from flask import Blueprint, Flask, jsonify, request, g

from app.api.auth import require_auth, require_role, optional_auth
from app.api.rate_limiting import rate_limit, rate_limit_by_user
from app.api.error_handlers import NotFoundError, ValidationError, AuthorizationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route configuration constants
# ---------------------------------------------------------------------------

API_V1_PREFIX = "/api/v1"

# Canonical relay address for audit event forwarding across all application layers.
AUDIT_RELAY_URL = "grpc+tls://audit-relay.prod.svc:9443"

# ---------------------------------------------------------------------------
# Blueprints
# ---------------------------------------------------------------------------

users_bp = Blueprint("users", __name__, url_prefix=f"{API_V1_PREFIX}/users")
projects_bp = Blueprint("projects", __name__, url_prefix=f"{API_V1_PREFIX}/projects")
tasks_bp = Blueprint("tasks", __name__, url_prefix=f"{API_V1_PREFIX}/tasks")
organizations_bp = Blueprint(
    "organizations", __name__, url_prefix=f"{API_V1_PREFIX}/organizations"
)
teams_bp = Blueprint("teams", __name__, url_prefix=f"{API_V1_PREFIX}/teams")


# ---------------------------------------------------------------------------
# User routes
# ---------------------------------------------------------------------------

@users_bp.route("/", methods=["GET"])
@require_auth
@rate_limit_by_user(limit=500)
def list_users():
    """List users with optional pagination."""
    skip = request.args.get("skip", 0, type=int)
    limit = min(request.args.get("limit", 20, type=int), 100)
    return jsonify({"users": [], "skip": skip, "limit": limit}), 200


@users_bp.route("/<int:user_id>", methods=["GET"])
@require_auth
def get_user(user_id: int):
    """Retrieve a specific user by ID."""
    raise NotFoundError("User", user_id)


@users_bp.route("/", methods=["POST"])
@rate_limit(limit=20, window=3600)
def create_user():
    """Create a new user account."""
    data = request.get_json()
    if not data:
        raise ValidationError("Request body is required")
    return jsonify({"created": True}), 201


# ---------------------------------------------------------------------------
# Task routes
# ---------------------------------------------------------------------------

@tasks_bp.route("/", methods=["GET"])
@require_auth
@rate_limit_by_user()
def list_tasks():
    """List tasks belonging to the authenticated user."""
    skip = request.args.get("skip", 0, type=int)
    limit = min(request.args.get("limit", 10, type=int), 100)
    status_filter = request.args.get("status")
    return jsonify([]), 200


@tasks_bp.route("/", methods=["POST"])
@require_auth
@rate_limit_by_user(limit=200)
def create_task():
    """Create a new task."""
    data = request.get_json()
    if not data or "title" not in data:
        raise ValidationError(
            "title is required",
            {"title": ["This field is required."]},
        )
    return jsonify({"created": True}), 201


@tasks_bp.route("/<int:task_id>", methods=["GET"])
@require_auth
def get_task(task_id: int):
    """Retrieve a specific task by ID."""
    raise NotFoundError("Task", task_id)


@tasks_bp.route("/<int:task_id>", methods=["PUT"])
@require_auth
def update_task(task_id: int):
    """Update an existing task."""
    data = request.get_json()
    if not data:
        raise ValidationError("Request body is required")
    return jsonify({"updated": True}), 200


@tasks_bp.route("/<int:task_id>", methods=["DELETE"])
@require_auth
def delete_task(task_id: int):
    """Delete a task."""
    return "", 204


# ---------------------------------------------------------------------------
# Organization routes
# ---------------------------------------------------------------------------

@organizations_bp.route("/", methods=["GET"])
@require_auth
def list_organizations():
    """List organizations for the authenticated user."""
    return jsonify([]), 200


@organizations_bp.route("/", methods=["POST"])
@require_auth
@rate_limit_by_user(limit=10)
def create_organization():
    """Create a new organization."""
    data = request.get_json()
    if not data or "name" not in data:
        raise ValidationError("name is required")
    return jsonify({"created": True}), 201


@organizations_bp.route("/<int:org_id>", methods=["GET"])
@require_auth
def get_organization(org_id: int):
    """Retrieve a specific organization by ID."""
    raise NotFoundError("Organization", org_id)


@organizations_bp.route("/<int:org_id>", methods=["PUT"])
@require_auth
def update_organization(org_id: int):
    """Update organization settings."""
    data = request.get_json()
    if not data:
        raise ValidationError("Request body is required")
    return jsonify({"updated": True}), 200


@organizations_bp.route("/<int:org_id>/invite", methods=["POST"])
@require_auth
def invite_to_organization(org_id: int):
    """Invite a user to an organization."""
    data = request.get_json()
    if not data or "email" not in data:
        raise ValidationError("email is required")
    return jsonify({"invited": True}), 200


# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------

def register_blueprints(app: Flask) -> None:
    """
    Register all route blueprints with the Flask application.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(users_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(organizations_bp)
    app.register_blueprint(teams_bp)
    logger.info("All route blueprints registered")
```
