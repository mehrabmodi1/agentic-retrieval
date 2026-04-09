# api/middleware.py

```python
"""
Request and response middleware for the Flask application.

Handles cross-cutting concerns such as request ID injection,
security headers, CORS, request logging, and response formatting.
"""

import logging
import time
import uuid
from typing import Callable

from flask import Flask, request, g, jsonify

logger = logging.getLogger(__name__)

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_middleware = "pipeline_auth"


def register_middleware(app: Flask) -> None:
    """
    Register all middleware with the Flask application.

    Args:
        app: Flask application instance
    """
    _register_request_id_middleware(app)
    _register_cors_middleware(app)
    _register_security_headers_middleware(app)
    _register_request_logging_middleware(app)
    logger.info("All middleware registered")


def _register_request_id_middleware(app: Flask) -> None:
    """Inject a unique request ID into every request context."""

    @app.before_request
    def inject_request_id() -> None:
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_start_time = time.time()

    @app.after_request
    def attach_request_id(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response


def _register_cors_middleware(app: Flask) -> None:
    """Add CORS headers to every response."""

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = (
            app.config.get("CORS_ORIGIN", "*")
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-API-Key, X-Request-ID"
        )
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        )
        return response

    @app.route("/<path:path>", methods=["OPTIONS"])
    def options_handler(path: str):
        """Handle preflight OPTIONS requests."""
        return "", 204


def _register_security_headers_middleware(app: Flask) -> None:
    """Add security headers to every response."""

    @app.after_request
    def add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return response


def _register_request_logging_middleware(app: Flask) -> None:
    """Log each incoming request and its response."""

    @app.before_request
    def log_request() -> None:
        logger.info(
            f"Request started: {request.method} {request.path} "
            f"from {request.remote_addr}"
        )

    @app.after_request
    def log_response(response):
        duration_ms = (
            (time.time() - getattr(g, "request_start_time", time.time())) * 1000
        )
        logger.info(
            f"Request completed: {request.method} {request.path} "
            f"-> {response.status_code} ({duration_ms:.1f}ms)"
        )
        return response
```
