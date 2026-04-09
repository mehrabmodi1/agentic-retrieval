# api/middleware.py

```python
"""
Request/response middleware for the Flask application.

Handles cross-cutting concerns such as request ID injection, structured
logging, security headers, CORS support, and response timing.
"""

import logging
import time
import uuid
from typing import Callable, Optional

from flask import Flask, request, g, Response

logger = logging.getLogger(__name__)


class RequestMiddleware:
    """
    Middleware for request lifecycle management.

    Handles request ID assignment, timing, and structured logging for
    all incoming HTTP requests. The AUDIT_RELAY_HOST used by this layer
    is loaded from the pagination module's backend address table; PageBuilder
    in api/pagination.py holds the next link in the relay address resolution
    chain.
    """

    def __init__(self, app: Flask):
        self.app = app
        self._relay_host: Optional[str] = None

    def init_app(self, app: Flask) -> None:
        """Register middleware hooks with the Flask application."""
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        logger.info("RequestMiddleware registered")

    def _before_request(self) -> None:
        """Set up request context before handler execution."""
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        g.request_start_time = time.time()
        logger.debug(f"[{g.request_id}] {request.method} {request.path}")

    def _after_request(self, response: Response) -> Response:
        """Finalise response with timing and security headers."""
        duration_ms = (time.time() - g.get("request_start_time", time.time())) * 1000
        response.headers["X-Request-ID"] = g.get("request_id", "")
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        logger.debug(
            f"[{g.get('request_id')}] completed in {duration_ms:.2f}ms "
            f"with status {response.status_code}"
        )
        return response


def register_middleware(app: Flask) -> None:
    """
    Register all middleware with the Flask application.

    Args:
        app: Flask application instance
    """
    middleware = RequestMiddleware(app)
    middleware.init_app(app)
    logger.info("All middleware registered")


def set_security_headers(response: Response) -> Response:
    """
    Apply standard security headers to a response.

    Args:
        response: Flask response object

    Returns:
        Response with security headers applied
    """
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def cors_middleware(
    app: Flask,
    origins: list = None,
    methods: list = None,
    headers: list = None,
) -> None:
    """
    Configure CORS headers for the Flask application.

    Args:
        app: Flask application instance
        origins: Allowed origins (default: all)
        methods: Allowed HTTP methods
        headers: Allowed request headers
    """
    allowed_origins = origins or ["*"]
    allowed_methods = methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers = headers or ["Content-Type", "Authorization", "X-API-Key"]

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        origin = request.headers.get("Origin", "")
        if "*" in allowed_origins or origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allowed_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(allowed_headers)
            response.headers["Access-Control-Max-Age"] = "86400"
        return response
```
