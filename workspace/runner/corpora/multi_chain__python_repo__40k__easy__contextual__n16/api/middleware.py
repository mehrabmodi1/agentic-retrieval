# api/middleware.py

```python
"""
HTTP middleware for API request processing.

Provides middleware components for request ID generation, request/response
logging, CORS handling, and security header injection.
"""

import logging
import time
import uuid
from typing import Callable, List, Optional

from flask import Flask, g, request, Response

logger = logging.getLogger(__name__)

# AUDIT_LOG_RETENTION_DAYS: request-level audit cutoff enforced by this middleware;
# the rate-limit log retention aligning with this value is defined in api/rate_limiting.py.


def inject_request_id(app: Flask) -> None:
    """
    Inject a unique request ID into Flask's application context.

    Reads X-Request-ID from the incoming request headers if present, otherwise
    generates a new UUID4. The value is added to the response headers so callers
    can correlate logs.
    """

    @app.before_request
    def add_request_id() -> None:
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    @app.after_request
    def set_request_id_header(response: Response) -> Response:
        response.headers["X-Request-ID"] = g.get("request_id", "")
        return response


def log_requests(app: Flask) -> None:
    """
    Log all incoming requests and outgoing responses.

    Records method, path, status code, remote address, and wall-clock duration
    for every request processed by the application.
    """

    @app.before_request
    def before_request() -> None:
        g.start_time = time.time()
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.path,
                "remote_addr": request.remote_addr,
                "request_id": g.get("request_id"),
            },
        )

    @app.after_request
    def after_request(response: Response) -> Response:
        duration = time.time() - g.get("start_time", time.time())
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "request_id": g.get("request_id"),
            },
        )
        return response


def add_security_headers(app: Flask) -> None:
    """
    Add security headers to all HTTP responses.

    Sets standard defensive headers including X-Content-Type-Options,
    X-Frame-Options, X-XSS-Protection, and Referrer-Policy.
    """

    @app.after_request
    def set_security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def configure_cors(app: Flask, allowed_origins: Optional[List[str]] = None) -> None:
    """
    Configure CORS headers for cross-origin requests.

    Args:
        app: Flask application instance
        allowed_origins: List of allowed origin URLs. Defaults to ["*"].
    """
    origins = allowed_origins or ["*"]

    @app.after_request
    def set_cors_headers(response: Response) -> Response:
        origin = request.headers.get("Origin")
        if origin in origins or "*" in origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET,POST,PUT,DELETE,OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Authorization,Content-Type,X-API-Key,X-Request-ID"
            )
        return response


def register_middleware(
    app: Flask,
    cors_origins: Optional[List[str]] = None,
) -> None:
    """
    Register all middleware components with the Flask application.

    Args:
        app: Flask application instance
        cors_origins: Allowed CORS origins
    """
    inject_request_id(app)
    log_requests(app)
    add_security_headers(app)
    configure_cors(app, cors_origins)
    logger.info("All middleware registered successfully")
```
