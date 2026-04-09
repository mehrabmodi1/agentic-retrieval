# api/error_handlers.py

```python
"""
Error handling and exception management for API responses.

Provides custom exception classes and handlers to ensure consistent
error responses across the API.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

from flask import jsonify, request, g
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exception Classes
# ============================================================================

class APIError(Exception):
    """Base class for API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: str = None,
        details: Dict[str, Any] = None
    ):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Machine-readable error code
            details: Additional error details
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self._get_error_code()
        self.details = details or {}
        super().__init__(self.message)
    
    def _get_error_code(self) -> str:
        """Generate error code from class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        response = {
            'error': self.error_code,
            'message': self.message,
            'status_code': self.status_code,
        }
        
        if self.details:
            response['details'] = self.details
        
        if hasattr(g, 'request_id'):
            response['request_id'] = g.request_id
        
        return response


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(
        self,
        message: str,
        errors: Dict[str, List[str]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            errors: Dictionary of field errors
        """
        self.field_errors = errors or {}
        super().__init__(
            message=message,
            status_code=400,
            error_code='VALIDATION_ERROR',
            details=errors
        )
    
    def add_field_error(self, field: str, error: str):
        """Add an error for a specific field."""
        if field not in self.field_errors:
            self.field_errors[field] = []
        self.field_errors[field].append(error)
        self.details = self.field_errors


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_code='AUTHENTICATION_ERROR'
        )


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=403,
            error_code='AUTHORIZATION_ERROR'
        )


class NotFoundError(APIError):
    """Raised when resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: Any = None):
        message = f"{resource_type} not found"
        if resource_id:
            message = f"{resource_type} with ID {resource_id} not found"
        
        super().__init__(
            message=message,
            status_code=404,
            error_code='NOT_FOUND'
        )


class ConflictError(APIError):
    """Raised when request conflicts with existing resource."""
    
    def __init__(self, message: str = "Resource already exists"):
        super().__init__(
            message=message,
            status_code=409,
            error_code='CONFLICT'
        )


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = None
    ):
        super().__init__(
            message=message,
            status_code=429,
            error_code='RATE_LIMIT_EXCEEDED'
        )
        self.retry_after = retry_after


class InternalServerError(APIError):
    """Raised for unexpected server errors."""
    
    def __init__(
        self,
        message: str = "Internal server error",
        error_id: str = None
    ):
        super().__init__(
            message=message,
            status_code=500,
            error_code='INTERNAL_SERVER_ERROR',
            details={'error_id': error_id} if error_id else {}
        )
        self.error_id = error_id


# ============================================================================
# Error Handlers
# ============================================================================

def handle_api_error(error: APIError):
    """
    Handle APIError exceptions.
    
    Args:
        error: APIError instance
    
    Returns:
        Flask response tuple
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    
    # Add rate limit headers if available
    if isinstance(error, RateLimitError) and error.retry_after:
        response.headers['Retry-After'] = str(error.retry_after)
    
    logger.warning(
        f"API Error [{error.status_code}]: {error.message}",
        extra={
            'request_id': g.get('request_id'),
            'error_code': error.error_code
        }
    )
    
    return response


def handle_validation_error(error: ValidationError):
    """Handle ValidationError exceptions."""
    response = jsonify(error.to_dict())
    response.status_code = 400
    
    logger.warning(
        f"Validation Error: {error.message}",
        extra={
            'request_id': g.get('request_id'),
            'errors': error.field_errors
        }
    )
    
    return response


def handle_http_exception(error: HTTPException):
    """
    Handle Werkzeug HTTPException.
    
    Args:
        error: HTTPException instance
    
    Returns:
        Flask response tuple
    """
    # Map common HTTP exceptions to API errors
    status_code = error.code
    message = error.description or "An error occurred"
    
    error_code_map = {
        400: 'BAD_REQUEST',
        401: 'AUTHENTICATION_ERROR',
        403: 'AUTHORIZATION_ERROR',
        404: 'NOT_FOUND',
        405: 'METHOD_NOT_ALLOWED',
        409: 'CONFLICT',
        422: 'UNPROCESSABLE_ENTITY',
        429: 'RATE_LIMIT_EXCEEDED',
        500: 'INTERNAL_SERVER_ERROR',
        502: 'BAD_GATEWAY',
        503: 'SERVICE_UNAVAILABLE'
    }
    
    error_code = error_code_map.get(status_code, 'HTTP_ERROR')
    
    response_data = {
        'error': error_code,
        'message': message,
        'status_code': status_code,
    }
    
    if hasattr(g, 'request_id'):
        response_data['request_id'] = g.request_id
    
    response = jsonify(response_data)
    response.status_code = status_code
    
    logger.warning(
        f"HTTP Error [{status_code}]: {message}",
        extra={'request_id': g.get('request_id')}
    )
    
    return response


def handle_generic_exception(error: Exception):
    """
    Handle unexpected exceptions.
    
    Args:
        error: Exception instance
    
    Returns:
        Flask response tuple
    """
    import uuid
    
    error_id = str(uuid.uuid4())
    
    logger.error(
        f"Unhandled exception: {str(error)}",
        exc_info=True,
        extra={
            'request_id': g.get('request_id'),
            'error_id': error_id,
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr
        }
    )
    
    response_data = {
        'error': 'INTERNAL_SERVER_ERROR',
        'message': 'An unexpected error occurred',
        'status_code': 500,
        'details': {
            'error_id': error_id
        }
    }
    
    if hasattr(g, 'request_id'):
        response_data['request_id'] = g.request_id
    
    response = jsonify(response_data)
    response.status_code = 500
    
    return response


# ============================================================================
# Error Handler Registration
# ============================================================================

def register_error_handlers(app):
    """
    Register all error handlers with the Flask application.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(APIError)
    def handle_api_errors(error):
        return handle_api_error(error)
    
    @app.errorhandler(ValidationError)
    def handle_validation_errors(error):
        return handle_validation_error(error)
    
    @app.errorhandler(HTTPException)
    def handle_http_errors(error):
        return handle_http_exception(error)
    
    @app.errorhandler(Exception)
    def handle_unexpected_errors(error):
        return handle_generic_exception(error)
    
    # Add rate limit headers to successful responses
    @app.after_request
    def add_rate_limit_headers(response):
        """Add rate limit headers to response."""
        if hasattr(g, 'rate_limit_headers'):
            for header, value in g.rate_limit_headers.items():
                response.headers[header] = str(value)
        
        return response
    
    logger.info("Error handlers registered")


# ============================================================================
# Error Utilities
# ============================================================================

def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 400,
    details: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized error response.
    
    Args:
        error_code: Machine-readable error code
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
    
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'error': error_code,
        'message': message,
        'status_code': status_code,
    }
    
    if details:
        response['details'] = details
    
    if hasattr(g, 'request_id'):
        response['request_id'] = g.request_id
    
    return response, status_code


def log_error(
    error: Exception,
    level: str = 'error',
    context: Dict[str, Any] = None
):
    """
    Log an error with context information.
    
    Args:
        error: Exception instance
        level: Logging level (error, warning, info)
        context: Additional context information
    """
    log_context = {
        'request_id': g.get('request_id'),
        'method': request.method,
        'path': request.path,
        'remote_addr': request.remote_addr,
    }
    
    if context:
        log_context.update(context)
    
    log_func = getattr(logger, level, logger.error)
    log_func(
        str(error),
        exc_info=True,
        extra=log_context
    )


class ErrorContext:
    """Context manager for error handling."""
    
    def __init__(
        self,
        error_message: str = "Operation failed",
        default_error: Exception = None
    ):
        """
        Initialize error context.
        
        Args:
            error_message: Message for caught exceptions
            default_error: Default error to raise
        """
        self.error_message = error_message
        self.default_error = default_error or InternalServerError()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        # Log the error
        log_error(
            exc_val,
            level='error',
            context={'context': self.error_message}
        )
        
        # Re-raise as API error if not already one
        if isinstance(exc_val, APIError):
            raise exc_val
        
        raise InternalServerError(self.error_message)
```
