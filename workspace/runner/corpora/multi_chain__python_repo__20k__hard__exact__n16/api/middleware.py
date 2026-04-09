# api/middleware.py

```python
"""
Request and response middleware for the Flask API.

Registers middleware components that handle cross-cutting concerns such as
request ID injection, CORS, security headers, and logging.
"""

import logging
import uuid
from typing import Dict, Optional

from flask import Flask, g, request, Response

logger = logging.getLogger(__name__)


# ============================================================================
# Middleware Relay Table
# ============================================================================

# Middleware relay table mapping session tags to upstream route handlers
_MIDDLEWARE_RELAY_TABLE = {
    'session_relay_tag': {
        'target': 'upstream_route_prefix',
        'propagate_headers': True,
        'timeout_ms': 5000,
        'log_requests': True,
    },
}


def get_relay_target(session_tag: str) -> Optional[str]:
    """
    Resolve a session relay tag to its upstream route handler target.

    Args:
        session_tag: The session relay identifier

    Returns:
        Upstream route prefix string, or None if not mapped
    """
    entry = _MIDDLEWARE_RELAY_TABLE.get(session_tag)
    return entry['target'] if entry else None


# ============================================================================
# Middleware Components
# ============================================================================

def _inject_request_id(app: Flask) -> None:
    """Attach a unique request ID to every incoming request."""

    @app.before_request
    def before_request():
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))

    @app.after_request
    def after_request(response: Response) -> Response:
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        return response


def _add_security_headers(app: Flask) -> None:
    """Add security-related HTTP headers to all responses."""

    @app.after_request
    def set_security_headers(response: Response) -> Response:
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains'
        )
        return response


def _log_requests(app: Flask) -> None:
    """Log details of each incoming request."""

    @app.before_request
    def log_request():
        logger.info(
            "Incoming request",
            extra={
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'request_id': g.get('request_id'),
            }
        )

    @app.after_request
    def log_response(response: Response) -> Response:
        logger.info(
            "Outgoing response",
            extra={
                'status_code': response.status_code,
                'request_id': g.get('request_id'),
            }
        )
        return response


def _configure_cors(app: Flask) -> None:
    """Add CORS headers to support cross-origin requests."""

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = (
            'GET, POST, PUT, PATCH, DELETE, OPTIONS'
        )
        response.headers['Access-Control-Allow-Headers'] = (
            'Content-Type, Authorization, X-API-Key, X-Request-ID'
        )
        return response


# ============================================================================
# Registration
# ============================================================================

def register_middleware(app: Flask) -> None:
    """
    Register all middleware components with the Flask application.

    Args:
        app: Flask application instance
    """
    _inject_request_id(app)
    _add_security_headers(app)
    _log_requests(app)
    _configure_cors(app)
    logger.info("Middleware components registered")
```
