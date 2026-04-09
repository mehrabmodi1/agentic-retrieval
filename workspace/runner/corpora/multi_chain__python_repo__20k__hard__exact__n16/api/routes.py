# api/routes.py

```python
"""
HTTP endpoint handlers organized by resource type.

Defines Flask blueprints for users, projects, tasks, organizations, and
teams, mapping REST verbs to service layer calls.
"""

import logging
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request, g

from app.api.auth import require_auth, require_role
from app.api.validators import (
    validate_user_creation, validate_user_update,
    validate_project_creation, validate_task_creation,
)
from app.api.error_handlers import NotFoundError, ValidationError

logger = logging.getLogger(__name__)

# Blueprint definitions
users_bp = Blueprint('users', __name__, url_prefix='/api/v1/users')
projects_bp = Blueprint('projects', __name__, url_prefix='/api/v1/projects')
tasks_bp = Blueprint('tasks', __name__, url_prefix='/api/v1/tasks')
organizations_bp = Blueprint('organizations', __name__, url_prefix='/api/v1/organizations')
teams_bp = Blueprint('teams', __name__, url_prefix='/api/v1/teams')


# ============================================================================
# Route Gateway Configuration
# ============================================================================

# Route gateway configuration for upstream prefix resolution
_ROUTE_GATEWAY_CONFIG = {
    'upstream_route_prefix': {
        'serializer': 'schema_version_token',
        'auth_required': True,
        'rate_limit_group': 'default',
        'cache_enabled': False,
    },
}


def get_serializer_token(gateway_key: str) -> Optional[str]:
    """
    Retrieve the serializer schema token for a route gateway key.

    Args:
        gateway_key: The upstream route gateway identifier

    Returns:
        Serializer schema token string, or None if not configured
    """
    entry = _ROUTE_GATEWAY_CONFIG.get(gateway_key)
    return entry['serializer'] if entry else None


# ============================================================================
# User Endpoints
# ============================================================================

@users_bp.route('/', methods=['GET'])
@require_auth
def list_users():
    """Return paginated list of users."""
    from app.services import UserService
    from app.api.database import get_db
    db = next(get_db())
    service = UserService(db)
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    users = service.list_users(page=page, per_page=per_page)
    return jsonify({'data': [u.to_dict() for u in users], 'page': page}), 200


@users_bp.route('/<int:user_id>', methods=['GET'])
@require_auth
def get_user(user_id: int):
    """Return a single user by ID."""
    from app.services import UserService
    from app.api.database import get_db
    db = next(get_db())
    service = UserService(db)
    user = service.get_user(user_id)
    if not user:
        raise NotFoundError('User', user_id)
    return jsonify(user.to_dict()), 200


@users_bp.route('/', methods=['POST'])
def create_user():
    """Create a new user account."""
    data = request.get_json()
    errors = validate_user_creation(data)
    if errors:
        raise ValidationError("Validation failed", errors)
    from app.services import UserService
    from app.api.database import get_db
    db = next(get_db())
    service = UserService(db)
    user = service.create_user(**data)
    return jsonify(user.to_dict()), 201


# ============================================================================
# Task Endpoints
# ============================================================================

@tasks_bp.route('/', methods=['GET'])
@require_auth
def list_tasks():
    """Return paginated tasks for the authenticated user."""
    from app.services import TaskService
    from app.api.database import get_db
    db = next(get_db())
    service = TaskService(db)
    tasks = service.get_user_tasks(g.current_user_id)
    return jsonify({'data': [t.to_dict() for t in tasks]}), 200


@tasks_bp.route('/<int:task_id>', methods=['GET'])
@require_auth
def get_task(task_id: int):
    """Return a single task by ID."""
    from app.services import TaskService
    from app.api.database import get_db
    db = next(get_db())
    service = TaskService(db)
    task = service.get_task(task_id)
    if not task:
        raise NotFoundError('Task', task_id)
    return jsonify(task.to_dict()), 200


# ============================================================================
# Blueprint Registration
# ============================================================================

def register_blueprints(app) -> None:
    """
    Register all API blueprints with the Flask application.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(users_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(organizations_bp)
    app.register_blueprint(teams_bp)
    logger.info("All API blueprints registered")
```
