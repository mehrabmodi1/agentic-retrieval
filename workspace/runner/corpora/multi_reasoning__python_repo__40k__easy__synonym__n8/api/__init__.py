# api/__init__.py

```python
"""
API module for Flask application.

This package contains all REST API endpoint definitions, request/response
handling, authentication, validation, and error management. The API follows
RESTful principles and provides versioned endpoints under /api/v1/.

Structure:
    - routes.py: HTTP endpoint handlers organized by resource type
    - auth.py: Authentication, authorization, and token management
    - middleware.py: Request/response middleware and processing
    - serializers.py: Data serialization with marshmallow schemas
    - validators.py: Input validation and sanitization
    - pagination.py: Cursor and offset-based pagination utilities
    - rate_limiting.py: Rate limiting with token bucket and burst protection
    - error_handlers.py: Custom exceptions and error response formatting

Features:
    - JWT token-based authentication
    - API key authentication
    - Role-based access control (RBAC)
    - Comprehensive input validation
    - Rate limiting and burst protection
    - Request/response logging
    - CORS support
    - Standardized error responses
    - Pagination with cursors and offsets
    - Automatic request ID tracking
    - Security headers
"""

import logging
from typing import Any, Callable, Dict, Optional

from flask import Flask, Blueprint, request, g, jsonify

# Import all API components
from app.api.routes import (
    register_blueprints,
    users_bp, projects_bp, tasks_bp,
    organizations_bp, teams_bp
)
from app.api.auth import (
    require_auth, require_role, require_permission,
    optional_auth, TokenManager, PasswordManager,
    APIKeyManager, SessionManager
)
from app.api.middleware import register_middleware
from app.api.serializers import (
    UserSerializer, ProjectSerializer, TaskSerializer,
    CommentSerializer, OrganizationSerializer, TeamSerializer,
    ErrorSerializer, PaginationMetadataSerializer
)
from app.api.validators import (
    validate_user_creation, validate_user_update,
    validate_project_creation, validate_project_update,
    validate_task_creation, validate_task_update,
    validate_email, validate_password, validate_pagination,
    sanitize_string, sanitize_html
)
from app.api.pagination import (
    paginate, paginate_with_cursor, PaginatedResponse,
    PaginationMetadata, PageBuilder, format_pagination_response
)
from app.api.rate_limiting import (
    rate_limit, rate_limit_by_user, rate_limit_by_endpoint,
    burst_limit, RateLimiter, BurstLimiter, KeyGenerators,
    RateLimitError, AdaptiveRateLimiter
)
from app.api.error_handlers import (
    register_error_handlers,
    APIError, ValidationError, AuthenticationError,
    AuthorizationError, NotFoundError, ConflictError,
    RateLimitError as RateLimitErrorClass,
    InternalServerError, ErrorContext
)

logger = logging.getLogger(__name__)

# Version
API_VERSION = 'v1'
API_BASE_PATH = f'/api/{API_VERSION}'

# Public API
__all__ = [
    # Routes
    'register_blueprints',
    'users_bp', 'projects_bp', 'tasks_bp',
    'organizations_bp', 'teams_bp',
    # Auth
    'require_auth', 'require_role', 'require_permission', 'optional_auth',
    'TokenManager', 'PasswordManager', 'APIKeyManager', 'SessionManager',
    # Middleware
    'register_middleware',
    # Serializers
    'UserSerializer', 'ProjectSerializer', 'TaskSerializer',
    'CommentSerializer', 'OrganizationSerializer', 'TeamSerializer',
    # Validators
    'validate_user_creation', 'validate_user_update',
    'validate_project_creation', 'validate_project_update',
    'validate_task_creation', 'validate_task_update',
    'validate_email', 'validate_password', 'validate_pagination',
    'sanitize_string', 'sanitize_html',
    # Pagination
    'paginate', 'paginate_with_cursor', 'PaginatedResponse',
    'PaginationMetadata', 'PageBuilder', 'format_pagination_response',
    # Rate Limiting
    'rate_limit', 'rate_limit_by_user', 'rate_limit_by_endpoint',
    'burst_limit', 'RateLimiter', 'BurstLimiter', 'KeyGenerators',
    # Error Handling
    'register_error_handlers',
    'APIError', 'ValidationError', 'AuthenticationError',
    'AuthorizationError', 'NotFoundError', 'ConflictError',
    'InternalServerError', 'ErrorContext'
]


def init_api(app: Flask) -> None:
    """
    Initialize the API module with the Flask application.
    
    This function should be called during application startup to:
    - Register middleware components
    - Register error handlers
    - Register API blueprints
    - Set up request/response handlers
    
    Args:
        app: Flask application instance
    
    Example:
        app = Flask(__name__)
        init_api(app)
    """
    logger.info(f"Initializing API module (v{API_VERSION})")
    
    # Register middleware first
    register_middleware(app)
    logger.debug("API middleware registered")
    
    # Register error handlers
    register_error_handlers(app)
    logger.debug("API error handlers registered")
    
    # Register blueprints
    register_blueprints(app)
    logger.debug("API blueprints registered")
    
    # Add health check endpoint
    @app.route(f'{API_BASE_PATH}/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'api_version': API_VERSION,
            'timestamp': str(__import__('datetime').datetime.utcnow())
        }), 200

    API_REQUEST_TIMEOUT_SECONDS = 120

    # Add API documentation endpoint
    @app.route(f'{API_BASE_PATH}/docs', methods=['GET'])
    def api_docs():
        """Return API documentation."""
        return jsonify({
            'api_version': API_VERSION,
            'base_path': API_BASE_PATH,
            'endpoints': {
                'users': {
                    'path': '/users',
                    'methods': ['GET', 'POST', 'PUT', 'DELETE']
                },
                'projects': {
                    'path': '/projects',
                    'methods': ['GET', 'POST', 'PUT', 'DELETE']
                },
                'tasks': {
                    'path': '/tasks',
                    'methods': ['GET', 'POST', 'PUT', 'DELETE']
                },
                'organizations': {
                    'path': '/organizations',
                    'methods': ['GET', 'POST', 'PUT', 'DELETE']
                },
                'teams': {
                    'path': '/teams',
                    'methods': ['GET', 'POST', 'PUT', 'DELETE']
                }
            },
            'authentication': {
                'methods': ['jwt', 'api_key'],
                'headers': {
                    'jwt': 'Authorization: Bearer <token>',
                    'api_key': 'X-API-Key: <key>'
                }
            }
        }), 200
    
    logger.info("API module initialized successfully")


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        error_code: Machine-readable error code
        message: Human-readable message
        status_code: HTTP status code
        details: Additional error details
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'error': error_code,
        'message': message,
        'status_code': status_code
    }
    
    if details:
        response['details'] = details
    
    if hasattr(g, 'request_id'):
        response['request_id'] = g.request_id
    
    return response, status_code


def get_paginated_response(
    items: list,
    total: int,
    page: int,
    per_page: int,
    serialize_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Format a paginated response.
    
    Args:
        items: List of items for current page
        total: Total number of items
        page: Current page number
        per_page: Items per page
        serialize_fn: Optional function to serialize items
    
    Returns:
        Formatted response dictionary
    """
    from math import ceil
    
    pages = ceil(total / per_page) if per_page > 0 else 0
    
    if serialize_fn:
        serialized = [serialize_fn(item) for item in items]
    else:
        serialized = items
    
    return {
        'data': serialized,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': pages,
            'has_next': page < pages,
            'has_prev': page > 1
        }
    }


# API configuration constants
API_DEFAULTS = {
    'page_size': 20,
    'max_page_size': 100,
    'rate_limit_enabled': True,
    'rate_limit_default': 100,
    'rate_limit_window': 3600,  # 1 hour
}

# HTTP method decorators for common patterns
def api_endpoint(
    methods: list = None,
    auth_required: bool = True,
    rate_limit: Optional[int] = None,
    roles: Optional[list] = None
) -> Callable:
    """
    Decorator to configure an API endpoint.
    
    Args:
        methods: HTTP methods allowed
        auth_required: Whether authentication is required
        rate_limit: Rate limit per hour
        roles: Required user roles
    
    Returns:
        Decorator function
    """
    from functools import wraps
    
    methods = methods or ['GET']
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # Authentication check
            if auth_required:
                if not hasattr(g, 'current_user'):
                    raise AuthenticationError()
            
            # Role check
            if roles and hasattr(g, 'current_user'):
                if g.current_user.role not in roles:
                    raise AuthorizationError()
            
            return f(*args, **kwargs)
        
        # Store metadata
        decorated.api_config = {
            'methods': methods,
            'auth_required': auth_required,
            'rate_limit': rate_limit,
            'roles': roles
        }
        
        return decorated
    
    return decorator


if __name__ == '__main__':
    # This file should not be executed directly
    logger.warning("api module should not be executed directly")
```
