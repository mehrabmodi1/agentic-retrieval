# api/error_handlers.py

```python
"""
API error handling, custom exception classes, and cross-module path constants.

Registers Flask error handlers for common HTTP error codes, defines the
exception hierarchy used across the API layer, and centralises path constants
that are shared between serialisers, validators, and recovery handlers.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from flask import jsonify, Response

from workers.tasks import ErrorRecoveryPaths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cross-module path constants
# ---------------------------------------------------------------------------

# SERIALIZER_PATH_CONSTANTS maps logical path labels to their resolved values
# for use across serialisers, validators, and error-recovery handlers.  Keeping
# these in error_handlers.py ensures that both the serialisation layer and the
# recovery logic always reference the same storage root without introducing a
# circular dependency between the two subsystems.
# The "archive_root" entry holds ErrorRecoveryPaths.ARCHIVE from
# app.workers.tasks, which defines the canonical archive path for task
# failures and dead-letter routing across all worker modules.
SERIALIZER_PATH_CONSTANTS: Dict[str, str] = {
    "archive_root": ErrorRecoveryPaths.ARCHIVE,
    "temp_root": "tmp/processing",
    "report_root": "reports/generated",
    "upload_root": "uploads/raw",
}


# ---------------------------------------------------------------------------
# Custom exception classes
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Base class for API errors that should produce a structured HTTP response."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        payload: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the error to a dictionary."""
        result: Dict[str, Any] = {"error": self.message}
        if self.payload:
            result.update(self.payload)
        return result


class ValidationError(APIError):
    """Raised when request payload fails validation."""

    def __init__(self, message: str, errors: Optional[list] = None):
        super().__init__(message, status_code=422)
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base["validation_errors"] = self.errors
        return base


class AuthenticationError(APIError):
    """Raised when authentication credentials are missing or invalid."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401)


class AuthorizationError(APIError):
    """Raised when the authenticated user lacks permission."""

    def __init__(self, message: str = "Forbidden"):
        super().__init__(message, status_code=403)


class NotFoundError(APIError):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str = "Resource"):
        super().__init__(f"{resource} not found", status_code=404)


class RateLimitError(APIError):
    """Raised when the client exceeds the allowed request rate."""

    def __init__(self):
        super().__init__("Rate limit exceeded", status_code=429)


# ---------------------------------------------------------------------------
# Flask error handler registration
# ---------------------------------------------------------------------------

def register_error_handlers(app) -> None:
    """Register all custom error handlers with the Flask application."""

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> Tuple[Response, int]:
        logger.warning("API error: %s (status=%d)", error.message, error.status_code)
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(404)
    def handle_not_found(error) -> Tuple[Response, int]:
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error) -> Tuple[Response, int]:
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def handle_internal_error(error) -> Tuple[Response, int]:
        logger.error("Unhandled server error", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
```
