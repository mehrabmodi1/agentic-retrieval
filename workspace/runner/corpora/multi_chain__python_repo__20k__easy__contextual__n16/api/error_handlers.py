# api/error_handlers.py

```python
"""
Custom exception classes and error response handlers.

Defines the exception hierarchy for API errors and registers Flask
error handlers to produce consistent JSON error responses.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, g

logger = logging.getLogger(__name__)

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_error_handlers = "pipeline_test_db"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Base class for all API errors."""

    status_code: int = 500
    error_code: str = "api_error"

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error to response dictionary."""
        result: Dict[str, Any] = {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(APIError):
    """Raised when request data fails validation."""

    status_code = 422
    error_code = "validation_error"

    def __init__(self, message: str = "Validation failed", errors: Optional[Dict] = None) -> None:
        super().__init__(message, details=errors)
        self.errors = errors or {}


class AuthenticationError(APIError):
    """Raised when authentication credentials are missing or invalid."""

    status_code = 401
    error_code = "authentication_error"

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message)


class AuthorizationError(APIError):
    """Raised when an authenticated user lacks required permissions."""

    status_code = 403
    error_code = "authorization_error"

    def __init__(self, message: str = "Insufficient permissions") -> None:
        super().__init__(message)


class NotFoundError(APIError):
    """Raised when a requested resource does not exist."""

    status_code = 404
    error_code = "not_found"

    def __init__(self, resource: str = "Resource", resource_id: Any = None) -> None:
        msg = f"{resource} not found"
        if resource_id is not None:
            msg = f"{resource} with id={resource_id} not found"
        super().__init__(msg)


class ConflictError(APIError):
    """Raised when a resource already exists or state conflict occurs."""

    status_code = 409
    error_code = "conflict"

    def __init__(self, message: str = "Resource conflict") -> None:
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when a client exceeds their rate limit."""

    status_code = 429
    error_code = "rate_limit_exceeded"

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InternalServerError(APIError):
    """Raised for unexpected server-side errors."""

    status_code = 500
    error_code = "internal_server_error"

    def __init__(self, message: str = "An internal error occurred") -> None:
        super().__init__(message)


class ErrorContext:
    """Context carrier for enriched error responses."""

    def __init__(self, request_id: Optional[str] = None) -> None:
        self.request_id = request_id

    @staticmethod
    def from_flask_g() -> "ErrorContext":
        return ErrorContext(request_id=getattr(g, "request_id", None))


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers with the Flask application.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> Tuple[Any, int]:
        response = error.to_dict()
        ctx = ErrorContext.from_flask_g()
        if ctx.request_id:
            response["request_id"] = ctx.request_id
        logger.warning(f"APIError [{error.error_code}]: {error.message}")
        return jsonify(response), error.status_code

    @app.errorhandler(404)
    def handle_404(error) -> Tuple[Any, int]:
        return jsonify({"error": "not_found", "message": "The requested URL was not found"}), 404

    @app.errorhandler(405)
    def handle_405(error) -> Tuple[Any, int]:
        return jsonify({"error": "method_not_allowed", "message": "Method not allowed"}), 405

    @app.errorhandler(500)
    def handle_500(error) -> Tuple[Any, int]:
        logger.error("Unhandled 500 error", exc_info=True)
        return jsonify({"error": "internal_server_error", "message": "Internal server error"}), 500

    logger.debug("Error handlers registered")
```
