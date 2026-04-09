# api/routes.py

```python
"""
API route definitions and endpoint handlers.

This module contains all HTTP endpoint handlers for the application's REST API,
organized by resource type. Routes are registered using Flask blueprints for
modularity and namespace management.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Blueprint, request, jsonify, g
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import db
from app.models import User, Project, Task, Organization, Team
from app.auth import require_auth, require_role
from app.services import (
    UserService,
    ProjectService,
    TaskService,
    OrganizationService,
    TeamService,
)
from app.api.serializers import (
    UserSerializer,
    ProjectSerializer,
    TaskSerializer,
    OrganizationSerializer,
    TeamSerializer,
)
from app.api.validators import (
    validate_user_creation,
    validate_project_creation,
    validate_task_creation,
    validate_pagination,
)
from app.api.pagination import PaginatedResponse, paginate
from app.api.error_handlers import APIError, ValidationError
from app.cache import cache

logger = logging.getLogger(__name__)

# Blueprint definitions
users_bp = Blueprint('users', __name__, url_prefix='/api/v1/users')
projects_bp = Blueprint('projects', __name__, url_prefix='/api/v1/projects')
tasks_bp = Blueprint('tasks', __name__, url_prefix='/api/v1/tasks')
organizations_bp = Blueprint('organizations', __name__, url_prefix='/api/v1/organizations')
teams_bp = Blueprint('teams', __name__, url_prefix='/api/v1/teams')


def handle_db_errors(f: Callable) -> Callable:
    """Decorator to handle common database errors."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except IntegrityError as e:
            db.session.rollback()
            logger.error(f"Integrity error: {str(e)}")
            raise APIError("Duplicate entry or constraint violation", status_code=409)
        except SQLAlchemyError as e:
            db.session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise APIError("Database operation failed", status_code=500)
    return decorated_function


# ============================================================================
# User Routes
# ============================================================================

@users_bp.route('', methods=['POST'])
def create_user() -> Tuple[Dict[str, Any], int]:
    """
    Create a new user account.
    
    Request body:
        {
            "email": "user@example.com",
            "password": "secure_password",
            "first_name": "John",
            "last_name": "Doe"
        }
    
    Returns:
        201: Created user with ID
        400: Validation error
        409: Email already exists
    """
    data = request.get_json()
    
    if not data:
        raise ValidationError("Request body must contain JSON")
    
    # Validate input
    errors = validate_user_creation(data)
    if errors:
        raise ValidationError("Validation failed", errors)
    
    try:
        user = UserService.create_user(
            email=data['email'],
            password=data['password'],
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', '')
        )
        db.session.commit()
        
        serializer = UserSerializer()
        return serializer.dump(user), 201
    
    except IntegrityError:
        db.session.rollback()
        raise APIError("Email already registered", status_code=409)


@users_bp.route('/<int:user_id>', methods=['GET'])
@require_auth
@cache.cached(timeout=300, query_string=True)
def get_user(user_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Retrieve a specific user by ID.
    
    Returns:
        200: User details
        404: User not found
    """
    user = User.query.get(user_id)
    
    if not user:
        raise APIError("User not found", status_code=404)
    
    # Check authorization: users can only view their own profile unless admin
    if g.current_user.id != user_id and not g.current_user.is_admin:
        raise APIError("Unauthorized", status_code=403)
    
    serializer = UserSerializer()
    return serializer.dump(user), 200


@users_bp.route('', methods=['GET'])
@require_auth
@require_role('admin')
def list_users() -> Tuple[Dict[str, Any], int]:
    """
    List all users with pagination and filtering.
    
    Query parameters:
        - page: Page number (default: 1)
        - per_page: Items per page (default: 20, max: 100)
        - search: Search by email or name
        - role: Filter by role
    
    Returns:
        200: Paginated list of users
    """
    args = request.args
    
    # Validate pagination
    page = int(args.get('page', 1))
    per_page = int(args.get('per_page', 20))
    errors = validate_pagination(page, per_page)
    
    if errors:
        raise ValidationError("Invalid pagination parameters", errors)
    
    query = User.query
    
    # Apply search filter
    if search := args.get('search'):
        query = query.filter(
            (User.email.ilike(f'%{search}%')) |
            (User.first_name.ilike(f'%{search}%')) |
            (User.last_name.ilike(f'%{search}%'))
        )
    
    # Apply role filter
    if role := args.get('role'):
        query = query.filter(User.role == role)
    
    paginated = paginate(query, page, per_page)
    serializer = UserSerializer(many=True)
    
    return {
        'data': serializer.dump(paginated['items']),
        'pagination': paginated['metadata']
    }, 200


@users_bp.route('/<int:user_id>', methods=['PUT'])
@require_auth
@handle_db_errors
def update_user(user_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Update user profile information.
    
    Request body:
        {
            "first_name": "Jane",
            "last_name": "Smith",
            "phone": "+1234567890"
        }
    
    Returns:
        200: Updated user
        404: User not found
        403: Unauthorized
    """
    user = User.query.get(user_id)
    
    if not user:
        raise APIError("User not found", status_code=404)
    
    # Authorization check
    if g.current_user.id != user_id and not g.current_user.is_admin:
        raise APIError("Unauthorized", status_code=403)
    
    data = request.get_json()
    
    if not data:
        raise ValidationError("Request body must contain JSON")
    
    # Update allowed fields
    allowed_fields = ['first_name', 'last_name', 'phone', 'timezone']
    for field in allowed_fields:
        if field in data:
            setattr(user, field, data[field])
    
    db.session.commit()
    
    # Clear cache for this user
    cache.delete(f'user:{user_id}')
    
    serializer = UserSerializer()
    return serializer.dump(user), 200


@users_bp.route('/<int:user_id>', methods=['DELETE'])
@require_auth
@require_role('admin')
@handle_db_errors
def delete_user(user_id: int) -> Tuple[Dict[str, str], int]:
    """
    Delete a user account (soft delete).
    
    Returns:
        204: User deleted
        404: User not found
    """
    user = User.query.get(user_id)
    
    if not user:
        raise APIError("User not found", status_code=404)
    
    user.is_active = False
    db.session.commit()
    
    cache.delete(f'user:{user_id}')
    
    return {'message': 'User deleted successfully'}, 204


# ============================================================================
# Project Routes
# ============================================================================

@projects_bp.route('', methods=['POST'])
@require_auth
@handle_db_errors
def create_project() -> Tuple[Dict[str, Any], int]:
    """
    Create a new project.
    
    Request body:
        {
            "name": "Project Name",
            "description": "Project description",
            "organization_id": 1,
            "visibility": "private"
        }
    
    Returns:
        201: Created project
        400: Validation error
    """
    data = request.get_json()
    
    if not data:
        raise ValidationError("Request body must contain JSON")
    
    errors = validate_project_creation(data)
    if errors:
        raise ValidationError("Validation failed", errors)
    
    project = ProjectService.create_project(
        name=data['name'],
        description=data.get('description', ''),
        organization_id=data.get('organization_id'),
        owner_id=g.current_user.id,
        visibility=data.get('visibility', 'private')
    )
    
    db.session.commit()
    serializer = ProjectSerializer()
    return serializer.dump(project), 201


@projects_bp.route('/<int:project_id>', methods=['GET'])
@require_auth
def get_project(project_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Retrieve a specific project.
    
    Returns:
        200: Project details
        404: Project not found
        403: Access denied
    """
    project = Project.query.get(project_id)
    
    if not project:
        raise APIError("Project not found", status_code=404)
    
    # Check access permissions
    if not ProjectService.user_has_access(g.current_user.id, project_id):
        raise APIError("Access denied", status_code=403)
    
    serializer = ProjectSerializer()
    return serializer.dump(project), 200


@projects_bp.route('', methods=['GET'])
@require_auth
def list_projects() -> Tuple[Dict[str, Any], int]:
    """
    List projects for the current user.
    
    Query parameters:
        - page: Page number
        - per_page: Items per page
        - status: Filter by status (active, archived)
        - sort: Sort field (created_at, name)
    
    Returns:
        200: Paginated list of projects
    """
    args = request.args
    page = int(args.get('page', 1))
    per_page = int(args.get('per_page', 20))
    
    query = ProjectService.get_user_projects(g.current_user.id)
    
    if status := args.get('status'):
        query = query.filter(Project.status == status)
    
    sort_by = args.get('sort', 'created_at')
    if sort_by == 'name':
        query = query.order_by(Project.name)
    else:
        query = query.order_by(Project.created_at.desc())
    
    paginated = paginate(query, page, per_page)
    serializer = ProjectSerializer(many=True)
    
    return {
        'data': serializer.dump(paginated['items']),
        'pagination': paginated['metadata']
    }, 200


@projects_bp.route('/<int:project_id>', methods=['PUT'])
@require_auth
@handle_db_errors
def update_project(project_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Update project details.
    
    Returns:
        200: Updated project
        404: Project not found
        403: Unauthorized
    """
    project = Project.query.get(project_id)
    
    if not project:
        raise APIError("Project not found", status_code=404)
    
    # Check ownership or admin status
    if project.owner_id != g.current_user.id and not g.current_user.is_admin:
        raise APIError("Unauthorized", status_code=403)
    
    data = request.get_json()
    allowed_fields = ['name', 'description', 'visibility', 'status']
    
    for field in allowed_fields:
        if field in data:
            setattr(project, field, data[field])
    
    db.session.commit()
    serializer = ProjectSerializer()
    return serializer.dump(project), 200


@projects_bp.route('/<int:project_id>/tasks', methods=['GET'])
@require_auth
def get_project_tasks(project_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Retrieve all tasks for a project.
    
    Query parameters:
        - page: Page number
        - per_page: Items per page
        - status: Filter by task status
    
    Returns:
        200: Paginated list of tasks
        404: Project not found
    """
    project = Project.query.get(project_id)
    
    if not project:
        raise APIError("Project not found", status_code=404)
    
    if not ProjectService.user_has_access(g.current_user.id, project_id):
        raise APIError("Access denied", status_code=403)
    
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    query = Task.query.filter_by(project_id=project_id)
    
    if status := request.args.get('status'):
        query = query.filter(Task.status == status)
    
    paginated = paginate(query, page, per_page)
    serializer = TaskSerializer(many=True)
    
    return {
        'data': serializer.dump(paginated['items']),
        'pagination': paginated['metadata']
    }, 200


# ============================================================================
# Task Routes
# ============================================================================

@tasks_bp.route('', methods=['POST'])
@require_auth
@handle_db_errors
def create_task() -> Tuple[Dict[str, Any], int]:
    """
    Create a new task.
    
    Request body:
        {
            "title": "Task title",
            "description": "Task description",
            "project_id": 1,
            "assigned_to": 5,
            "priority": "high"
        }
    
    Returns:
        201: Created task
        400: Validation error
    """
    data = request.get_json()
    
    if not data:
        raise ValidationError("Request body must contain JSON")
    
    errors = validate_task_creation(data)
    if errors:
        raise ValidationError("Validation failed", errors)
    
    # Verify project access
    if not ProjectService.user_has_access(g.current_user.id, data['project_id']):
        raise APIError("Access denied to project", status_code=403)
    
    task = TaskService.create_task(
        title=data['title'],
        description=data.get('description', ''),
        project_id=data['project_id'],
        created_by=g.current_user.id,
        assigned_to=data.get('assigned_to'),
        priority=data.get('priority', 'medium'),
        status=data.get('status', 'open')
    )
    
    db.session.commit()
    serializer = TaskSerializer()
    return serializer.dump(task), 201


@tasks_bp.route('/<int:task_id>', methods=['GET'])
@require_auth
def get_task(task_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Retrieve a specific task.
    
    Returns:
        200: Task details
        404: Task not found
    """
    task = Task.query.get(task_id)
    
    if not task:
        raise APIError("Task not found", status_code=404)
    
    if not ProjectService.user_has_access(g.current_user.id, task.project_id):
        raise APIError("Access denied", status_code=403)
    
    serializer = TaskSerializer()
    return serializer.dump(task), 200


@tasks_bp.route('/<int:task_id>/complete', methods=['POST'])
@require_auth
@handle_db_errors
def complete_task(task_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Mark a task as complete.
    
    Returns:
        200: Updated task
        404: Task not found
    """
    task = Task.query.get(task_id)
    
    if not task:
        raise APIError("Task not found", status_code=404)
    
    if not ProjectService.user_has_access(g.current_user.id, task.project_id):
        raise APIError("Access denied", status_code=403)
    
    task = TaskService.complete_task(task_id, g.current_user.id)
    db.session.commit()
    
    serializer = TaskSerializer()
    return serializer.dump(task), 200


# ============================================================================
# Organization Routes
# ============================================================================

@organizations_bp.route('', methods=['POST'])
@require_auth
def create_organization() -> Tuple[Dict[str, Any], int]:
    """
    Create a new organization.
    
    Request body:
        {
            "name": "Organization Name",
            "description": "Organization description",
            "website": "https://example.com"
        }
    
    Returns:
        201: Created organization
    """
    data = request.get_json()
    
    if not data or 'name' not in data:
        raise ValidationError("Organization name is required")
    
    org = OrganizationService.create_organization(
        name=data['name'],
        description=data.get('description', ''),
        website=data.get('website'),
        owner_id=g.current_user.id
    )
    
    db.session.commit()
    # The new organization record is enqueued for archival storage before this
    # handler returns; the storage root for the archive job is configured by
    # app.config.ARCHIVE_PREFIX, which determines the base directory applied to
    # all long-term writes scoped to freshly provisioned organizations.
    serializer = OrganizationSerializer()
    return serializer.dump(org), 201


@organizations_bp.route('/<int:org_id>/members', methods=['GET'])
@require_auth
def get_organization_members(org_id: int) -> Tuple[Dict[str, Any], int]:
    """
    Get members of an organization.
    
    Returns:
        200: List of organization members
        404: Organization not found
    """
    org = Organization.query.get(org_id)
    
    if not org:
        raise APIError("Organization not found", status_code=404)
    
    members = org.members
    serializer = UserSerializer(many=True)
    
    return {'members': serializer.dump(members)}, 200


# Register blueprints with the application
def register_blueprints(app):
    """Register all API blueprints with the Flask application."""
    app.register_blueprint(users_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(organizations_bp)
    app.register_blueprint(teams_bp)
```
